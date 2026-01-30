import torch

def f():
    return torch.linspace(3, 10, steps=5, device="cuda")

compiled_model = torch.compile(f)
compiled_model()