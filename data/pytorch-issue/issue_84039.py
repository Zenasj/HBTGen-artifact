from torch import einsum, ones
import argparse

parser = argparse.ArgumentParser(description='mpsndarray test')
parser.add_argument('--n_samples', type=int, default=2)
args = parser.parse_args()
n_samples = args.n_samples

einsum('b i d, b j d -> b i j', ones(16 * n_samples, 4096, 40, device='mps'), ones(16 * n_samples, 4096, 40, device='mps')).shape

print(n_samples, 'passed')

from torch import einsum, ones
# crashes with "product of dimension sizes > 2**31"
# this is equivalent to invoking stable-diffusion with --n_samples 2
einsum('b i d, b j d -> b i j', ones(32, 4096, 40, device='mps'), ones(32, 4096, 40, device='mps')).shape

# doesn't crash, even though it's bigger
# this is equivalent to invoking stable-diffusion with --n_samples 3
einsum('b i d, b j d -> b i j', ones(48, 4096, 40, device='mps'), ones(48, 4096, 40, device='mps')).shape

from torch import einsum, ones
import argparse

parser = argparse.ArgumentParser(description='mpsndarray test')
parser.add_argument('--n_samples', type=int, default=2)
args = parser.parse_args()
n_samples = args.n_samples

einsum('b i d, b j d -> b i j', ones(16 * n_samples, 4096, 40, device='mps'), ones(16 * n_samples, 4096, 40, device='mps')).shape

print(n_samples, 'passed')

# Crashes
t1 = torch.rand((32, 4096, 40))
t2 = torch.rand((32, 4096, 40))
torch.matmul(t1.to("mps"), t2.to("mps").transpose(1, 2)).shape

# Doesn't crash, even though it's bigger
t1 = torch.rand((48, 4096, 40))
t2 = torch.rand((48, 4096, 40))
torch.matmul(t1.to("mps"), t2.to("mps").transpose(1, 2)).shape

t1 = torch.rand((48, 4096, 40))
t2 = torch.rand((48, 4096, 40))

x_mps = einsum('b i d, b j d -> b i j', t1.to('mps'), t2.to('mps'))
x_cpu = einsum('b i d, b j d -> b i j', t1, t2)
print((x_mps.to("cpu") - x_cpu).abs().max())
# tensor(9.4567) !?

t1 = torch.rand((48, 4096, 40))
t2 = torch.rand((48, 4096, 40))

x_mm_mps = torch.matmul(t1.to("mps"), t2.to("mps").transpose(1, 2))
x_mm_cpu = torch.matmul(t1, t2.transpose(1, 2))
print((x_mm_mps.to("cpu") - x_mm_cpu).abs().max())
# tensor(0.)

x_mps = einsum('b i d, b j d -> b i j', t1.to('mps'), t2.to('mps'))
x_cpu = einsum('b i d, b j d -> b i j', t1, t2)
print((x_mps.to("cpu") - x_cpu).abs().max())
# tensor(0.)

import torch

t1 = torch.ones((32, 4096, 4096))
t2 = torch.ones((32, 4096, 1))
torch.matmul(t1.to("mps"), t2.to("mps")).shape

from diffusers import StableDiffusionPipeline

sdm = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    safety_checker=None,
).to("mps")

prompt = "A painting of a squirrel eating a burger"
num_samples = 2

images = sdm(prompt, num_images_per_prompt=num_samples).images
for i, image in enumerate(images):
    image.save(f"squirrel_{i}.png")

from diffusers import StableDiffusionPipeline

sdm = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    safety_checker=None,
).to("mps")

prompt = "A painting of a squirrel eating a burger"
num_samples = 2

images = sdm(prompt, num_images_per_prompt=num_samples, num_inference_steps=20).images
for i, image in enumerate(images):
    image.save(f"squirrel_{i}.png")