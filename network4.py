import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        
        self.attn = Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)))
        self.ff = Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
        
    def forward(self, x):
        x = self.attn(x)
        x = self.ff(x)
        return x


class net(nn.Module):
    def __init__(self, image_size=32, patch_size=4, dim=512, heads=8, mlp_dim=512, budget=10, channels = 3, dropout = 0.05, emb_dropout = 0.05):
        super().__init__()
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.delta = 1
        
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.module1 = SkipNet(dim, heads, mlp_dim, dropout, budget)
        self.max = budget
        
        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 10)
        )
        
    def forward(self, x, y):
        p = self.patch_size

        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        
        if self.training:
            x, halt = self.module1(x) # (B, max, 65, 512), (B, max)
            x = self.to_cls_token(x[:, :, 0]) # (B, max, 512)
            x = self.mlp_head(x) # (B, max, class)
            
            labels = repeat(y, 'b n -> b (l n)', l = self.max) # (B, 1) -> (B, max)
            logits = rearrange(x, 'b l d -> b d l') # (B, max, class) -> (B, class, max)
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            index = ce_loss.argmin(1, keepdim=False) + 1
            ref_ce_loss = torch.stack(list(map(lambda i: ce_loss[i, :index[i]].mean(), range(b))))
            
            ones = torch.ones((b, self.max)).to(x.device)
            halt_labels = list(map(lambda i: change(ones,i), enumerate(index)))
            halt_logits = list(map(lambda i: halt[i[0], :i[1]], enumerate(index)))
            halt_ce_loss = torch.stack(list(map(lambda x: F.binary_cross_entropy(x[0], x[1], reduction='mean'), zip(halt_logits, halt_labels))))
            
            loss = ref_ce_loss.mean() + self.delta * halt_ce_loss.mean()
        
            return loss
        
        else:
            x, cnt = self.module1(x) # (B, 65, 512)
            x = self.to_cls_token(x[:, 0]) # (B, 512)
            x = self.mlp_head(x) # (B, class)
            return x, cnt

def change(tensor, x):
    tensor[x[0],x[1]-1] = 0
    return tensor[x[0], :x[1]]

class SkipNet(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, budget):
        super().__init__()
        
        self.max = budget
        self.module = Transformer(dim, heads, mlp_dim, dropout)
        
        self.halt = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,1),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        outputs = []
        halts = []
        
        if self.training:
            for i in range(self.max):
                x = self.module(x)
                halt = self.halt(x).mean(dim=1).squeeze(1)
                outputs.append(x)
                halts.append(halt)
            
            outputs = torch.stack(outputs, dim = 1)
            halts = torch.stack(halts, dim = 1)
            return outputs, halts
        else:
            x = self.module(x)
            for i in range(self.max):
                halt = self.halt(x).mean(dim=1).squeeze(1)
                check = (halt >= 0.5)
                if check.sum() == 0 or (i+1) == self.max:
                    return x, (i+1)
                else:
                    x[check] = self.module(x[check])
    

def roop2(var):
    print(var, 'input')
    var = var + 0.1
    
    check = var < 0.5
    if check.sum() == 0:
        print(var, 'No F')
        return var
    else:
        var[check] = roop2(var[check])
        print(var, 'Re check')
        return var 
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B = 3
    a = torch.randn((B,3,32,32)).to(device).softmax(dim=-1)
    target = torch.randint(10, (B,1), dtype=torch.int64).to(device)
    b = net().to(device)
    
    loss = b(a, target)
    print(loss)
