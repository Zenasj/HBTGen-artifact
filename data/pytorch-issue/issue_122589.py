import torch

class TestTensor(torch.Tensor):
    pass

@torch.compile
def test(x):
    y = x.as_subclass(TestTensor)

x = torch.randn(2,4,3)

test(x)