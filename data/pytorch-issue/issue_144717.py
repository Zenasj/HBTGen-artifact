import torch
import einops
from torch._inductor import config as inductor_config
from torch._dynamo.testing import rand_strided, reset_rng_state

inductor_config.fallback_random = True

image_latent = torch.randn((24, 16, 32, 32), device="cuda").to(memory_format=torch.channels_last).view(2, 12, 16, 32, 32)

def f(image_latent):
    indices = torch.argsort(torch.rand(2, 12), dim=-1)[:, : 6]

    tar_latent = image_latent[
        torch.arange(2).unsqueeze(-1), indices[:, 3:]
    ]

    tar_latent_rearranged = einops.rearrange(
        tar_latent, "b n c h w -> (b n) c h w"
    )
    return {
        "tar_latent": tar_latent,
        "tar_latent_rearranged": tar_latent_rearranged,
    }


reset_rng_state()
ref = f(image_latent)
opt_f = torch.compile(f)
reset_rng_state()
act = opt_f(image_latent)
print(f"max dif {(act['tar_latent'] - ref['tar_latent']).abs().max()}")
print(f"max dif {(act['tar_latent_rearranged'] - ref['tar_latent_rearranged']).abs().max()}")