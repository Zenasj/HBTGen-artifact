import torch

batch = torch.arange(10).to("mps")
batch = [x for x in batch]
print(batch)
x = torch.stack(batch, 0, out=None)
print(x)