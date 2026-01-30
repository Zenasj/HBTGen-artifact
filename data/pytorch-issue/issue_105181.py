pipe = StableDiffusionXLPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

#...
for x in pipe.__module__:
    del x
del pipe
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

from diffusers import DiffusionPipeline
import torch, gc
#Any model will suffice for demonstration, this is from huggingface diffusers
pipe = DiffusionPipeline.from_pretrained(base_model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

#...
for x in pipe.__module__:
    del x
del pipe
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

for x in pipe.__module__:
    del x
del pipe
gc.collect()
torch._dynamo.reset()
torch.cuda.empty_cache()
torch.cuda.synchronize()