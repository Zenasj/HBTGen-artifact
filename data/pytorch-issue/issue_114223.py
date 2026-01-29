# torch.rand((), dtype=torch.float16)  # Scalar input tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model0(nn.Module):
    def __init__(self, v11_0):
        super().__init__()
        self.v11_0 = v11_0

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        abs_1 = torch.abs(self.v11_0)
        div = torch.div(getitem, abs_1)
        cat = torch.cat((div,), dim=2)
        transpose = div.transpose(2, 4)
        cat_1 = torch.cat((transpose,), dim=4)
        floor = torch.floor(cat_1)
        floor_1 = torch.floor(floor)
        mean = floor_1.mean(4)
        interpolate = F.interpolate(
            floor_1,
            size=[1, 2, 10],
            mode='trilinear',
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False
        )
        return (cat, mean, interpolate)

class Model1(nn.Module):
    def __init__(self, v11_0):
        super().__init__()
        self.v11_0 = v11_0

    def forward(self, *args):
        _args = args
        getitem = _args[0]
        abs_1 = torch.abs(self.v11_0)
        div = torch.div(getitem, abs_1)
        cat = torch.cat((div,), dim=2)
        transpose = div.transpose(2, 4)
        cat_1 = torch.cat((transpose,), dim=4)
        floor = torch.floor(cat_1)
        floor_1 = torch.floor(floor)
        mean = floor_1.mean(4)
        interpolate = F.interpolate(
            floor_1,
            size=[1, 2, 10],
            mode='trilinear',
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False
        )
        return (interpolate, cat, mean)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize shared parameter with random values to avoid division by zero
        self.v11_0 = nn.Parameter(
            torch.randn([57, 1, 10, 1, 40], dtype=torch.float16),
            requires_grad=False
        )
        self.model0 = Model0(self.v11_0)
        self.model1 = Model1(self.v11_0)

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        # Compare mean outputs (Model0's second element vs Model1's third element)
        are_close = torch.allclose(
            out0[1], out1[2], rtol=1.0, atol=0.0
        )
        return torch.tensor(are_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand((), dtype=torch.float16)

