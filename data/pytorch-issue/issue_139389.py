from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
import sys
import diffusers
import torch
import time


prompt = "Emily Booth as Sappho, cold color palette, vivid colors, detailed, 8k, 35mm photo, Kodachrome, Lomography, highly detailed"
negative_prompt = "painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured"

prompt="Nina Hagen as a 1960s (stingray puppet)++, supermarionation++ strings visible ,  ,  . vivid colors, detailed, 8k, 35mm photo, Kodachrome, Lomography, highly detailed"

negative_prompt="painting, drawing, illustration, glitch, mutated, cross-eyed, ugly, disfigured"

isteps=30
height=1024
width=1024
seed = 432773521

print(f"Python version {sys.version} PyTorch version {torch.__version__} diffusers.version is {diffusers.__version__}")

vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                    subfolder='vae',
                                    torch_dtype=torch.bfloat16,
                                    force_upcast=False).to('mps')


pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
    torch_dtype=torch.bfloat16, variant="fp16").to('mps')

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

generator =  torch.Generator("mps").manual_seed(seed);

start_time = time.time()
start_mps_mem = torch.mps.driver_allocated_memory()
image = pipe(prompt=prompt, negative_prompt=negative_prompt,
               height=height, width=width,
               num_inference_steps=10,
               guidance_scale=8,
               generator=generator,
               ).images[0]
end_mps_mem = torch.mps.driver_allocated_memory()
run_time = time.time() - start_time
print(f"run time in {run_time:.2f} sec, end_mps_mem {end_mps_mem/1024.0**2:.2f} Mb mem increase {(end_mps_mem-start_time)/1024.0**2:.2f} Mb")
image.save(f'bfloat16.png')