import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from unet_attention import SpatialTransformer
class UNetModel(nn.Module):
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
        
        """
            in_channels: the number of channels in the input feature map
            out_channels: the number of channels in the output feature map
            channels: the base channel count for the model
            n_res_blocks: number of residual blocks at each level
            attention_levels: the levels at which attention should be performed
            channel_multipliers: the multiplicative factors for number of channels for each level
            n_heads: the number of attention heads in the transformers
        """
        
        super().__init__()
        self.channels = channels
        
        
        levels = len(channel_multipliers)

        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, d_time_emb),
            nn.Silu(),
            nn.Linear(d_time_emb, d_time_emb),
        )
        
        
        # Input half
        self.input_blocks = nn.ModuleList()
        
        self.input_blocks.append(TimestepEmbedSequential(
            nn.Conv2d(in_channels, channels, 3, padding=1)))
        
        
        input_block_channels = [channels]
        
        channels_list = [channels * m for m in channel_multipliers]
        
        for i in range(levels):
            
            for _ in range(n_res_blocks):
                
                
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)

            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)
        
        # Middle half
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers, d_cond),
            ResBlock(channels, d_time_emb),
        )
        
        
        self.output_blocks = nn.ModuleList([])
        
        for i in reversed(range(levels)):
            
            for j in range(n_res_blocks + 1):
                
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                
                if i in attention_levels:
                    layers.append(SpatialTransformer(channels, n_heads, tf_layers, d_cond))
                

                if i!=0 and j==n_res_blocks:
                    layers.append(UpSample(channels))
                
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
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
        """
            x: [batch_size, channels, width, height]
            cond: [batch_size, n_cond, d_cond]
        
        """
        x_input_block = []
        
        
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)
        
        
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        
        x = self.middle_block(x, t_emb, cond)
        
        for module in self.output_blocks:
            x = module(torch.cat([x, x_input_block.pop()]), dim=1)
            x = module(x, t_emb, cond)
        
        return self.out(x)

class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor):
        # x: [batch_size, channels, height, width]
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor):
        return self.op(x)
        
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
            