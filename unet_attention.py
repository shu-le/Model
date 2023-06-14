from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    use_flash_attention: bool = False
    def __init__(self, d_model: int, d_cond: int, n_heads: int, d_head: int, is_inplace: bool = True):
        super.__init__()
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
        Args:
            x (torch.Tensor): [batch_size, height * width, d_model]
            cond (Optional[torch.Tensor], optional): the conditional embeddings of shape: [batch_size, n_cond, d_cond]
        """
        has_cond = cond is not None
        if not has_cond:
            cond = x
        
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        
    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        qkv: [batch_size, seq, d_attn]
        """
        q = q.view(*q.shape[:2], self.n_heads, -1)
        k = k.view(*k.shape[:2], self.n_heads, -1)
        v = v.view(*v.shape[:2], self.n_heads, -1)
        
        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale
        