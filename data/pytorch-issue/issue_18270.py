import torch

@torch.jit.script
def fn(x, y: Optional[torch.Tensor]=None, c:float=0.1):
  if y is not None:
      y.copy_(y * c + (1-c) * x.detach())
  return x

a = torch.randn(5,5, device='cuda', requires_grad=True)
b = torch.randn_like(a, requires_grad=False)
fn(a, b)
print (b.requires_grad)