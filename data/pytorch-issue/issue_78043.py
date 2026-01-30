import torch

generator = transformers.pipeline(task="text-generation", model=model.to('mps'), tokenizer=tokenizer, device=torch.device("mps"))
generator("This shall brake. ", max_length=200, use_cache=True)