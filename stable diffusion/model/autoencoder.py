from typing import List

import torch
import torch.nn.functional as F
from torch import nn

class Autoencoder(nn.Module):
    
    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', emb_channels: int, z_channels: int):
        """
        emb_channels is the number of dimensions in the quantized embedding space
        z_channels is the number of channels in the embedding space
        """
        
        super().__init__()
        self.edcoder = encoder
        self.decoder = decoder
        
        
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * emb_channels, 1)
        self.post_quant_conv = nn.Conv2d(2 * emb_channels, 2 * z_channels, 1)
        
    def encoder(self, img: torch.Tensor) -> 'GaussianDistribution':
        """
        img is the image tensor with shape [batch_size, img_channels, img_height, img_width]
        """
        
        # Get embeddings with shape [batch_size, z_channels * 2, z_height, z_height]
        z = self.encoder(img)
        
        moments = self.quant_conv(z)
        
        return GaussianDistribution(moments)
    
    def decoder(self, z: torch.Tensor):
        
        z = self.post_quant_conv(z)
        
        return self.decoder(z)


class Encoder(nn.Module):

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int, 
                 in_channels: int, z_channels: int)
        """
        channels is the number of channels in the first convolution layer
        channel_multipliers are the multiplicative factors for the number of channels in the subsequent blocks
        n_resnet_blocks is the number of resnet layers at each resolution
        in_channels is the number of channels in the image
        z_channels is the number of channels in the embedding space
        """
        super().__init__()
        
        n_resolutions = len(channel_multipliers)
        
        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)
        
        channels_list = [m * channels for m in [1] + channel_multipliers]
        
        self.down = nn.ModuleList()
        
        for i in range(n_resolutions):
            
            resnet_blocks = nn.ModuleList()
            
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]

            down = nn.Module()
            down.block = resnet_blocks
            
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
                
            self.down.append(down)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)
        
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)
    
    def forward(self, img: torch.Tensor):
        # img is the image tensor with shape [batch_size, img_channels, img_height, img_width]
        
        x = self.conv_in(img)
        
        for down in self.down:
            
            for block in down.block:
                x = block(x)
            
            x = down.downsample(x)
        
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)
        
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 out_channels: int, z_channels: int):
        """
        channels is the number of channels in the final convolution layer
        channel_multipliers are the multiplicative factors for the number of channels in the previous blocks, in reverse order
        n_resnet_blocks is the number of resnet layers at each resolution
        out_channels is the number of channels in the image
        z_channels is the number of channels in the embedding space
        """  
        super().__init__()
        
        n_resolutions = len(channel_multipliers)
        
        channels_list = [m * channels for m in channel_multipliers]
        
        channels = channels_list[-1]
        
        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)
        
        self.up = nn.ModuleList()
        
        for i in reversed(range(n_resolutions)):
            
            resnet_blocks = nn.ModuleList()
            
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
                
            up = nn.Module()
            up.block = resnet_blocks
            
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            
            self.up.insert(0, up)
        
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1)
        
    def forward(self, z: torch.Tensor):
        # z is the embedding tensor with shape [batch_size, z_channels, z_height, z_height]
        
        h = self.conv_in(z)
        
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        for up in reversed(self.up):
            
            for block in up.block:
                h = block(h)
            
            h = up.upsample(h)
        
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)
        
        return img
            

class GaussianDistribution:
    
    def __init__(self, parameters: torch.Tensor):
        """
        parameters are the means and log of variances of the embedding of shape [batch_size, z_channels * 2, z_height, z_height]
        """
        
        self.mean, log_var = torch.chunk(parameters, 2, dim=1)
        
        self.log_var = torch.clamp(log_var, -30.0, 20.0)
        
        self.std = torch.exp(0.5 * self.log_var)
        
    def sample(self):
        
        return self.mean + self.std * torch.randn_like(self.std)


class AttnBlock(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.norm = normalization(channels)
        
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        self.scale = channels ** -0.5
    
    def forward(self, x: torch.Tensor):
        # x is the tensor of shape [batch_size, channels, height, width]
        
        x_norm = self.norm(x)
        
        q = self.q(x.norm)
        k = self.k(x.norm)
        v = self.v(x.norm)
        
        b, c, h, w = q.shape
        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)
        
        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        
        out = torch.einsum('bij,bcj->bci', attn, v)
        
        out = out.view(b, c, h, w)
        
        out = self.proj_out(out)
        
        return x + out


class UpSample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.conv = nn.Conv(channels, channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor):
        # x is the input feature map with shape [batch_size, channels, height, width]

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        
        return self.conv(x)
        

class DownSample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 3×3 convolution with stride length of 2 to down-sample by a factor of 2
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)
    
    def forward(self, x: torch.Tensor):
        # x is the input feature map with shape [batch_size, channels, height, width]
        
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        
        return self.conv(x)    


class ResnetBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        x is the input feature map with shape [batch_size, channels, height, width]
        """
        
        h = x
        
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        
        return self.nin_shortcut(x) + h    
        

def swish(x: torch.Tensor):
    return x * torch.sigmoid(x)

def normalization(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
