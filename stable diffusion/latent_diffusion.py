from typing import List

import torch
import torch.nn as nn

from model.autoencoder import Autoencoder
from model.clip_embedder import CLIPTextEmbedder
from model.unet import UNetModel


class DiffusionWrapper(nn.Module):
    """
    
        This is an empty wrapper class around the U-Net. We keep this to have the same model 
        structure as CompVis/stable-diffusion so that we do not have to map the checkpoint 
        weights explicitly.
    """
    def __init__(self, diffusion_model: UNetModel):
        super().__init__()
        self.diffusion_model = diffusion_model
    
    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):
        return self.diffusion_model(x, time_steps, context)


class LatentDiffusion(nn.Module):
    
    model: DiffusionWrapper
    first_stage_model: UNetModel
    cond_stage_model: CLIPTextEmbedder
    
    def __init__(self, 
                 unet_model: UNetModel,
                 autoencode: Autoencoder,
                 clip_embedder: CLIPTextEmbedder,
                 latent_scaling_factor: float,
                 n_steps: int,
                 linear_start: float,
                 linear_end: float,
                 ):
        """

        unet_model is the U-Net that predicts noise ϵ(x, c) , in latent space
        autoencoder is the AutoEncoder
        clip_embedder is the CLIP embeddings generator
        latent_scaling_factor is the scaling factor for the latent space. The encodings of the autoencoder are scaled by this before feeding into the U-Net.
        n_steps is the number of diffusion steps T.
        linear_start is the start of the β schedule.
        linear_end is the end of the β schedule.
        """
        super().__init__()
        
        self.model = DiffusionWrapper(unet_model)
        
        self.first_stage_model = autoencode
        self.latent_scaling_factor = latent_scaling_factor
        
        self.cond_stage_model = clip_embedder
        
        
        self.n_steps = n_steps
        
        
        beta = torch.linspace(linear_start ** 2, linear_end ** 2, n_steps, dtype=torch.float64) ** 2
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)
        
        alpha = 1. - beta
        
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)
    
    @property
    def device(self):
        return next(iter(self.model.parameters())).device
    
    def get_text_conditioning(self, prompts: List[str]):
        # Get CLIP embeddings for a list of text prompts
        
        return self.cond_stage_model(prompts)
    
    def autoencoder_encode(self, image: torch.Tensor):
        # Get scaled latent space representation of the image
        # The encoder output is a distribution. We sample from that and multiply by the scaling factor.
        
        return self.latent_scaling_factor * self.first_stage_model.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        # Get image from the latent representation
        # scale down by the scaling factor and then decode.
        
        return self.first_stage_model.decode(z/self.latent_scaling_factor)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor):
        # Predict noise
        
        return self.model(x, t, context)