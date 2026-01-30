import torch.nn.functional as F

py
import torch.nn as nn
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

    def forward(self, input_tensor):
        x = self.conv(input_tensor)
        x = F.relu(x + torch.rand(x.size()))
        return x

batch_size = 1
x = torch.rand(batch_size, 3, 224, 224)

func = Model()

res1 = func(x)
print(res1)

with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)
    res2 = jit_func(x)
    print(res2)
    # AssertionError: the MutationLayout's real layout shouldn't be FlexibleLayout