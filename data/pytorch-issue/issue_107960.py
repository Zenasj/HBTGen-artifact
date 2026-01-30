import torch

torch.set_default_device("cuda:0")
@torch.compile
def test(x):
  return torch.sin(x)

a = torch.zeros(100)
test(a)