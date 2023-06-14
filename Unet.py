import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class UnetModel(nn.Module):
    """
    in_channels is the number of channels in the input feature map
    out_channels is the number of channels in the output feature map
    channels is the base channel count for the model
    n_res_blocks number of residual blocks at each level
    attention_levels are the levels at which attention should be performed
    channel_multipliers are the multiplicative factors for number of channels for each level
    n_heads the number of attention heads in the transformers
    """
    def __init__(self, *,
                 in_channels: int,
                 out_channels: int,
                 channels: int,
                 n_res_blocks: int,
                 attention_levels: List[int],
                 channel_multipliers: List[int],
                 n_heads: int,
                 tf_layers: int = 1,
                 d_cond: int = 768):
        
        super().__init__()
        self.channels = channels

        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.Silu(),
            nn.Linear(d_time_emb, d_time_emb),
        )
        
        
        
    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """Create sinusoidal time step embeddings
            cos(t/pow(10000, 2i/d)) for even i = 2, 4, 6, ..., d
            sin(t/pow(10000, 2i/d)) for odd i = 1, 3, 5, ..., d-1
        Args:
            time_steps (torch.Tensor): t
            max_period (int, optional): Defaults to 10000.
        """
        half = self.channels // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(device=time_steps.device)   # 1/10000^(2i/d)
        args = time_steps[:, None] * frequencies[None, :]  # t/pow(10000, 2i/d)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: torch.Tensor):
        
        t_emb = self.time_step_embedding(time_steps)
            
        
class ResBlock(nn.Module):
    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = channels
        
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding = 1),
        )
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels)
        )
        
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        
        if channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        """
        Args:
            x (torch.Tensor): the input feature map with shape [batch_size, channels, height, width]
            t_emb (torch.Tensor): the time step embeddings of shape [batch_size, d_t_emb]
        """
        h = self.in_layers(x)
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        h = h + t_emb[:, :, None, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h
        
        
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels):
    return GroupNorm32(32, channels)
            