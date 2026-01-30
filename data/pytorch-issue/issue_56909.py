import torch

def f(x): 
   return getattr(torch, "unknown", [])

torch.jit.script(f)