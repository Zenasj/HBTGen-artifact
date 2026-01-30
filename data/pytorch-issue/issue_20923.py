import torch

def foo(x):
    return ~x

y = torch.jit.trace(foo, torch.zeros(3,4, dtype=torch.uint8))
print(y.code)

def foo(x: Tensor) -> Tensor:
  _0 = torch.empty([3, 4], dtype=0, layout=0, device=torch.device("cpu"), pin_memory=False)
  _1 = torch.sub_(torch.fill_(_0, 1), x, alpha=1)
  return _1