import torch
@torch.compile()
def test():
    a = Tensor([i for i in range(10)])
    a[a > 5] = 0

test()

torch.where(a > 5, 0, a)

tensor[a > b]

self(x)

tensor[a>b]