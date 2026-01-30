py
import torch
import torch.nn as nn

set_seed(420)


class BinaryModel(nn.Module):

    def __init__(self):
        super(BinaryModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.add(x, x)
        return x
x = torch.randn(1, 3, 32, 32)

func = BinaryModel()

func.train(False)
with torch.no_grad():
    # without optimization
    res1 = func(x)
    print(res1)

    # with optimization
    # it triggered the fuse_binary optimization
    try:
        fn = torch.compile(func)
        res2 = fn(x)
    except Exception as e:
        print(e)