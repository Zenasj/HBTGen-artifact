import torch


from diffusers import (
   StableDiffusionXLPipeline,
   AutoencoderKL
)


# Load VAE component
vae = AutoencoderKL.from_pretrained(
   "madebyollin/sdxl-vae-fp16-fix",
   torch_dtype=torch.float16
)


# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
   "cagliostrolab/animagine-xl-3.0",
   vae=vae,
   torch_dtype=torch.float16,
   use_safetensors=True,
)
pipe.to('cuda')


# Define prompts and generate image
prompt = "1girl, arima kana, oshi no ko, solo, upper body, v, smile, looking at viewer, outdoors, night"
negative_prompt = "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, artist name"


pipe = torch.compile(pipe)


image = pipe(
   prompt,
   negative_prompt=negative_prompt,
   width=832,
   height=1216,
   guidance_scale=7,
   num_inference_steps=28
).images[0]