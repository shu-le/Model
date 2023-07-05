from typing import Optional, List

import numpy as np
import torch

from labml import monit
from ..latent_diffusion import LatentDiffusion
from . import DiffusionSampler

class DDPMSampler(DiffusionSampler):
    
    model: LatentDiffusion

    def __init__(self, model: LatentDiffusion):
        # model is the model to predict noise ϵ(x_t, c)
        super().__init__(model)
        
        self.time_steps = np.asarray(List(range(self.n_steps)))
        
        with torch.no_grad():
            
            alpha_bar = self.model.alpha_bar
            
            beta = self.model.beta
            
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])
            
            self.sqrt_alpha_bar = alpha_bar ** 0.5
            
            self.sqrt_1m_alpha_bar = (1. - alpha_bar) ** 0.5
            
            self.sqrt_recip_alpha_bar = alpha_bar ** -0.5
            
            self.sqrt_recip_1m_alpha_bar = (1 / alpha_bar - 1) ** 0.5
            
            variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
            
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            
            self.mean_x0_coef = beta * (alpha_bar_prev ** 0.5) / (1. - alpha_bar)
            
            self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1. - alpha_bar)
            
    @torch.no_grad()
    def sample(self,
            shape: List[int],
            cond: torch.Tensor,
            repeat_noise: bool = False,
            temperature: float = 1.,
            x_last: Optional[torch.Tensor] = None,
            uncond_scale: float = 1.,
            uncond_cond: Optional[torch.Tensor] = None,
            skip_steps: int = 0,
            ):  
        """
        shape [batch_size, channels, height, width]
        cond is the conditional embeddings c
        temperature is the noise temperature (random noise gets multiplied by this)
        x_last is x_T If not provided random noise will be used
        uncond_scale is the unconditional guidance scale s
        uncond_cond is the conditional embedding for empty prompt c_u
        skip_steps is the number of time steps to skip t', start from T-t', so x_last is x_T-t'
        """
        
        device = self.model.device
        bs = shape[0]
        
        # Get x_T
        x = x_last if x_last is not None else torch.randn(shape, device=device)
        
        # Time steps to sample at T-t',T-t'-1,...,1
        time_steps = np.flip(self.time_steps)[skip_steps:]
        
        for step in monit.iterate('Sample', time_steps):
            # batch个t
            ts = x.new_full((bs), step, dtype=torch.long)
            
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)
        
        return x
            
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None
                 ):   
        """
        x is x_t of shape [batch_size, channels, height, width]
        c is the conditional embeddings c of shape [batch_size, emb_size]
        t is t of shape [batch_size]
        step is the step t as an integer
        repeat_noise: specified whether the noise should be same for all samples in the batch
        temperature is the noise temperature (random noise gets multiplied by this)
        """
        # get ϵθ
        e_t = self.get_eps(x, t, c,
                           uncond_scale=uncond_scale,
                           uncond_cond=uncond_cond)
        
        bs = x.shape[0]
        
        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step])
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_1m_alpha_bar[step])
        
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t
        
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])
        
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])
        
        if step == 0:
            noise = 0
            
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]))
        
        else:
            noise = torch.randn(x.shape)
            
        noise = noise * temperature
        
        x_prev = mean + (0.5 * log_var).exp() * noise
        
        return x_prev, x0, e_t
        
    
    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        x0 is x_0 of shape [batch_size, channels, height, width]
        index is the time step t index
        noise is the noise ϵ
        """
        
        if noise is None:
            noise = torch.randn_like(x0)
        
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise