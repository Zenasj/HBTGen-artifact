import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",torch_dtype=torch.float16)
pipe = pipe.to("cuda")

generator = torch.Generator("cuda")
generator.manual_seed(123456)

image1 = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=32,
    guidance_scale=7.0,
    generator=generator
).images[0]
image2 = pipe(
    "A cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=32,
    guidance_scale=7.0,
    generator=generator
).images[0]
image1

image2