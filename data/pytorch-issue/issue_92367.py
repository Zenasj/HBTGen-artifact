import torch

def fn(x):
    return torch.Tensor(x)
x = [1, 2, 3]
torch._dynamo.optimize("eager")(fn)(x)

x = [1, 2, 3]
with torch._subclasses.fake_tensor.FakeTensorMode():
    y = torch.Tensor(x)