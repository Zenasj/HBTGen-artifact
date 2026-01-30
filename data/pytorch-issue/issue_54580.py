import torch
import torch.nn as nn
from copy import deepcopy

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.con1 = nn.Conv2d(1, 2, 3)
        self.con2 = nn.Conv2d(2, 1, 3)

    def forward(self, x):
        if True:
            return self.func(x)

    def func(self, x):
        self.exp = None
        x = self.con1(x)
        x = self.con2(x)
        self.exp = x
        return self.exp

model = test().cuda()
input = torch.Tensor(1,1,16,16)
output = model(input.cuda())
copyModel = deepcopy(model)
print("")