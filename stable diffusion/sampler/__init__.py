from typing import Optional, List
import torch
from ..latent_diffusion import LatentDiffusion

class DiffusionSampler:
    
    model: LatentDiffusion
    
    def __init__(self, model: LatentDiffusion):
        super().__init__()
        
        self.model = model
        self.n_steps = model.n_steps
        
    def get_eps(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, *,
                uncond_scale: float, uncond_cond: Optional[torch.Tensor]):
        """
        x shape [batch_size, channels, height, width]
        t shape [batch_size]
        c is the conditional embeddings c of shape [batch_size, emb_size]
        uncond_scale is the unconditional guidance scale s.
        uncond_cond is the conditional embedding for empty prompt c.
        """
        
        if uncond_cond is None or uncond_scale == 1:
            return self.model(x, t, c)
        
        
        x_in = torch.cat([x] * 2)
        t_in = torch.cat([t] * 2)
        
        c_in = torch.cat([uncond_cond, c])
        
        e_t_uncond, e_t_cond = self.model(x_in, t_in, c_in).chunk(2)
        
        # classifier-free guidance
        e_t = e_t_uncond + uncond_scale * (e_t_cond - e_t_uncond)
        
        return e_t
    
    # Sampling Loop
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
        skip_steps is the number of time steps to skip
        """
        
        raise NotImplementedError()

    # Painting Loop
    def paint(self, x: torch.Tensor, cond: torch.Tensor, t_start: int, *,
              orig: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None, orig_noise: Optional[torch.Tensor] = None,
              uncond_scale: float = 1.,
              uncond_cond: Optional[torch.Tensor] = None,
              ):
        """
        x is x_T' of shape [batch_size, channels, height, width]
        cond is the conditional embeddings c
        t_start is the sampling step to start from, T'
        orig is the original image in latent page which we are in painting
        mask is the mask to keep the original image
        orig_noise is fixed noise to be added to the original image
        uncond_scale is the unconditional guidance scale s
        uncond_cond is the conditional embedding for empty prompt c_u
        """
        
        raise NotImplementedError()
    
    # Sample from q(x_t|x_0)
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        x0 is x_0 of shape [batch_size, channels, height, width]
        index is the time step t index
        noise is the noise ϵ
        """
        
        raise NotImplementedError()
    