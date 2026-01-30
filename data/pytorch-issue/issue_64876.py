import torch

class MyTensor(torch.Tensor):
    pass

x = torch.ones(5)

x.grad = MyTensor(torch.ones(5))