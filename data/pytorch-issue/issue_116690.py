import torch.nn as nn

import torch
from diffusers import (
    PNDMScheduler,
    UNet2DConditionModel,
)
from torch.fx.experimental.proxy_tensor import make_fx


class Scheduler(torch.nn.Module):
    def __init__(self, hf_model_name, num_inference_steps):
        super().__init__()
        self.scheduler = PNDMScheduler.from_pretrained(hf_model_name, subfolder="scheduler")
        self.scheduler.set_timesteps(num_inference_steps)
        self.unet = UNet2DConditionModel.from_pretrained(
            hf_model_name,
            subfolder="unet",
        )
        self.guidance_scale = 7.5

    def forward(self, latents, encoder_hidden_states) -> torch.FloatTensor:
        latents = latents * self.scheduler.init_noise_sigma
        for t in self.scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep=t)
            unet_out = self.unet.forward(
                latent_model_input, t, encoder_hidden_states, return_dict=False
            )[0]
            noise_pred_uncond, noise_pred_text = unet_out.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        return latents


if __name__ == '__main__':
    hf_model_name = "CompVis/stable-diffusion-v1-4"
    scheduler = Scheduler(hf_model_name, 5)
    inputs = (torch.randn(1, 4, 64, 64), torch.randn(2, 77, 768),)

    fx_g = make_fx(
        scheduler,
        decomposition_table={},
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
        _allow_fake_constant=False,
    )(*inputs)

    print(fx_g)

import torch
from torch.fx.experimental.proxy_tensor import make_fx

def fn(a, timestep):
    alphas_cumprod = torch.cumprod(a, dim=0)
    set_alpha_to_one = False
    final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else alphas_cumprod[0]
    alpha_prod_t = torch.index_select(alphas_cumprod, 0, timestep)
    alpha_prod_t_prev = torch.index_select(alphas_cumprod, 0, timestep) if timestep >= 0 else final_alpha_cumprod
    #alpha_prod_t_prev = torch.where(timestep >= 0, torch.index_select(alphas_cumprod, 0, timestep), final_alpha_cumprod)
    return alpha_prod_t_prev

inputs = (torch.tensor([1, 4]), torch.tensor(1), )

fx_g = make_fx(
    fn,
    decomposition_table={},
    tracing_mode="symbolic",
    _allow_non_fake_inputs=True,
    _allow_fake_constant=False,
)(*inputs)

print(fx_g)

c = torch.scalar_tensor(c)