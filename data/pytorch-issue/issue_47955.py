py

import torch

x = torch.rand(10, 10)

print(x.sum(dim=0).shape)  # torch.Size([10])
print(x.sum(dim=0, keepdim=True).shape)  # torch.Size([1, 10])
print(x.sum(dim=0, keepdims=True).shape)  # torch.Size([1, 10])


# works
def test(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=0, keepdim=True)
torch.jit.script(test)


# fails
def tests(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=0, keepdims=True)
torch.jit.script(tests)