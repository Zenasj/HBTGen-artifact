import torch
x=torch.arange(32, device="mps")
x[::2].bitwise_not_()
print(x)