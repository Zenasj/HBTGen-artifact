import torch

a = torch.empty(10, dtype=torch.complex64).normal_()
torch.add(a, a, alpha=0.1)  # For integral input tensors, argument alpha must not be a floating point number.