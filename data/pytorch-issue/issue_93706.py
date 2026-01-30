import torch
def test_size(t):
    size = t.size()
    print(size)
    return size
out = torch.compile(test_size, backend="eager")(torch.randn(10, 10))