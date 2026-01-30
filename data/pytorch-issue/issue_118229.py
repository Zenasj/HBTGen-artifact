import torch
def f(x):
  return x.sin().element_size() + x.sin()

x = torch.randn(2, 2)
torch.compile(f, backend="eager", fullgraph=True)(x)