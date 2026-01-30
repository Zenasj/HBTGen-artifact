import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

@torch.compile(backend="inductor")
def inference_func(promt):
    image = pipe(prompt, num_inference_steps=1).images[0]
    return image

prompt = "a photo of an astronaut riding a horse on mars"
image = inference_func(prompt)