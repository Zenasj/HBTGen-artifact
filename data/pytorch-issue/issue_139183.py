import torch

@torch.compile(fullgraph=True)
def split(x):
    return x.split(4, 0)

x = torch.randn(12)
split(x)
print('cpu passed')
x = torch.randn(12, device="cuda")
split(x)
print('cuda passed')

with torch.device("cuda"):
    x = torch.randn(12)
    split(x) # breaks
    print('cuda cm passed')