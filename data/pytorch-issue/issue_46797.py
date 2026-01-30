import torch

x = torch.rand(8)
y = torch.rand(8)
z = torch.maximum(x, y, out=x)

def maximum(input: Tensor, other: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...