import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True) # Note: Change from `backend="cudagraphs" to `mode="reduce-overhead"`
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True) #  Note: Change from `backend="cudagraphs" to `mode="reduce-overhead"`
# pipe.text_encoder = torch.compile(pipe.text_encoder, fullgraph=True)
# pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, fullgraph=True)

n_steps = 4
# shapes = [[512, 512], [768, 768], [1024,1024]]
shapes = [[512, 512],]
batch_num = [1, 2 ,4 ,8]
prompt = "A majestic lion jumping from a big stone at night"

for i in range(len(shapes)):
    for j in range(len(batch_num)):
        torch.compiler.cudagraph_mark_step_begin()
        shape = shapes[i]

        #warmup
        for warmup in range(2):
            _ = pipe(
                prompt=prompt,
                width=shape[0],
                height=shape[1],
                num_inference_steps=n_steps,
                denoising_end=1,
                generator=torch.manual_seed(0),
                num_images_per_prompt=batch_num[j]
                ).images

        images = pipe(
            prompt=prompt,
            width=shape[0],
            height=shape[1],
            num_inference_steps=n_steps,
            denoising_end=1,
            generator=torch.manual_seed(0),
            num_images_per_prompt=batch_num[j]
            ).images
        print("shape = ", shape)
        print("image = ", batch_num[j])