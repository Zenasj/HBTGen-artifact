from diffusers import AutoPipelineForText2Image
import torch
import matplotlib.pyplot as plt
import time

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
guidance_scale = 7.5  # Set the guidance scale to your desired value

pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

start_time = time.time()
image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=guidance_scale).images[0]
end_time = time.time()
elapsed_time = end_time - start_time
gpu_memory_used = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB