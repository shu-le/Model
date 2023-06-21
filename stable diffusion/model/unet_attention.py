from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, n_layers: int, d_cond: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1,padding=0)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(channels, n_heads, channels//n_heads, d_cond=d_cond) for _ in range(n_layers)]
        )
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1,padding=0)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
            x (torch.Tensor): [batch_size, channels, height, width]
            cond (torch.Tensor): [batch_size, n_cond, d_cond]
        """
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.permute(0, 2, 3, 1).view(b, h*w, c)
        for block in self.transformer_blocks:
            x = block(x, cond)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.proj_out(x)
        return x + x_in
        

class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_cond: int):
        super.__init__()
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn2 = CrossAttention(d_model, d_cond, n_heads, d_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond=cond) + x
        x = self.ff(self.norm3(x)) + x
        
        return x


class CrossAttention(nn.Module):
    use_flash_attention: bool = False
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        super().__init__()
        self.is_inplace = is_inplace
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        d_attn = d_head * n_heads
        
        self.to_q = nn.Linear(d_model, d_attn, bias=False)
        self.to_k = nn.Linear(d_cond, d_attn, bias=False)
        self.to_v = nn.Linear(d_cond, d_attn, bias=False)
        self.to_out = nn.Sequential(nn.Linear(d_attn, d_model))
    
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
            x (torch.Tensor): [batch_size, height * width, d_model]
            cond (Optional[torch.Tensor], optional): the conditional embeddings of shape: [batch_size, n_cond, d_cond]
        """
        has_cond = cond is not None
        if not has_cond:
            cond = x
        
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        
        return self.normal_attention(q, k, v)
        
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
            qkv: [batch_size, seq, d_attn]
        """
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)
        
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
            
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = out.reshape(*out.shape[:2], -1)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_mult: int = 4):
        """
            d_mult: multiplicative factor for the hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            GeGLU(d_model, d_model * d_mult)
            nn.Dropout(0.)
            nn.Linear(d_model * d_mult, d_model)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)
        

class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out * 2)
    
    def forward(self, x: torch.Tensor):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)