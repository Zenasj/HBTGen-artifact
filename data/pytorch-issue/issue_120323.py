import torch
from diffusers import DiffusionPipeline

with torch._subclasses.FakeTensorMode():
    fake_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=False)