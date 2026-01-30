import torch.nn as nn

py
import torch
import torch.nn.functional as F

torch.manual_seed(420)

x = torch.randn(1, 3, 2, 2)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = torch.randn(3, 1, 1, 1)
        self.bias = torch.randn(3)
        self.bn = torch.nn.BatchNorm2d(num_features=3, affine=False)

    def forward(self, x):
        return F.batch_norm(x, self.bn.running_mean, self.bn.running_var, self.weight, self.bias)


func = Model().to('cpu')

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1.shape) # torch.Size([1, 3, 2, 2])

    res2 = jit_func(x)
    print(res2.shape) # torch.Size([3, 3, 2, 2])