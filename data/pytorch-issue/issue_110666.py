import torch

@torch.compile
def generate(x):
    with torch.no_grad():
        # assume model has some params with requires_grad=True
        return model(*x)