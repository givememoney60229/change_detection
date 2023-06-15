import torch
from torch import nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class tokenization(nn.Module):
    def __init__(self, pixel_num, pixel_num_b):
        super().__init__()
        self.block=nn.Sequential(
            nn.Linear(pixel_num, pixel_num_b),
            nn.GELU(),
       
           
            )
    def forward(self,x):
        x=x.squeeze()
        x=x.permute(0, 2, 1).reshape([ x.shape[0],-1])
        x=self.block(x).unsqueeze(0)

        return x

class detokenization(nn.Module):
    def __init__(self, pixel_num, pixel_num_b,image_size):
        super().__init__()
        self.width,self.height=image_size[0],image_size[1]
        self.block=nn.Sequential(
            nn.Linear(pixel_num_b, pixel_num),
            )
    def forward(self,x):
        x=x.squeeze()
        x=self.block(x)
        x=x.transpose(0,1).reshape(x.shape[0],self.height,self.width).permute(0,2,1).unsqueeze(0)

        return x

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
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
    
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # BNC -> BNH(C/H) -> BHN(C/H)
        q = self.wq(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BHN(C/H) @ BH(C/H)N -> BHNN
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (BHNN @ BHN(C/H)) -> BHN(C/H) -> BNH(C/H) -> BNC
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=3., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x[:, 1:, :]))) 
        x = x+ self.drop_path(self.attn(x)) # Better result
        if self.has_mlp:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttentionBlock(dim, num_heads=heads, drop=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x=attn(x)+self.norm(x)
            #x = torch.cat((attn(x), self.norm(x[:, 1:, :])), dim=1)
            x = ff(x) + x
        return x


class transformer_proj(nn.Module):
    def __init__(self,size, dim=5000, depth=2, heads=10, mlp_dim=5000, dropout=0.15):
        super().__init__()
        pixel_num=size[2]*size[3]
        self.token=tokenization(pixel_num,dim)
        self.transformer=Transformer(dim,depth,heads,mlp_dim,dropout=dropout)
        self.recon=detokenization(pixel_num,dim,(size[2],size[3]))
    def forward(self,x):
        x=self.token(x)
        x=self.transformer(x)
        x=self.recon(x)
        return x






if __name__ == "__main__":
    feature=torch.rand(1,201,382,256)
    dim=10000
    depth=3
    heads=10
    mlp_dim=dim
    size=(1,201,382,256)
    project=transformer_proj(size,dim=dim,depth=depth,heads=heads,mlp_dim=mlp_dim)
    feature2=project(feature)
    print(np.shape(feature2))
