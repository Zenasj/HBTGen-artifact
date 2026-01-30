import torch

@dynamo.optimize("tvm")
def toy_example(a, b):
    a = torch.cos(a)
    return a * torch.sin(b)


for _ in range(100):
    toy_example(torch.randn(10), torch.randn(10))