# torch.rand(1, 1, 1, 26, 1, dtype=torch.float16)
import torch
import torch.nn as nn

class Model0(torch.nn.Module):
    def __init__(self, v5_0):
        super().__init__()
        self.v5_0 = v5_0

    def forward(self, *args):
        getitem = args[0]
        mul = torch.mul(getitem, self.v5_0)
        softmax = torch.softmax(mul, dim=0)
        div = torch.div(getitem, softmax)
        return (div,)

class Model1(torch.nn.Module):
    def __init__(self, v5_0):
        super().__init__()
        self.v5_0 = v5_0

    def forward(self, *args):
        getitem = args[0]
        mul = torch.mul(getitem, self.v5_0)
        cat = torch.cat((mul, mul), dim=0)
        softmax = torch.softmax(mul, dim=0)
        div = torch.div(getitem, softmax)
        return (cat, div)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.nn.Parameter(
            torch.empty([3, 42, 7, 26, 1], dtype=torch.float16),
            requires_grad=False
        )
        self.model0 = Model0(self.p0)
        self.model1 = Model1(self.p0)

    def forward(self, x):
        out0 = self.model0(x)
        out1 = self.model1(x)
        div0 = out0[0]
        div1 = out1[1]
        # Comparison logic using rtol=1, atol=0
        mask = torch.abs(div0 - div1) <= (1 * torch.abs(div1))
        all_close = torch.all(mask)
        return torch.tensor([all_close], dtype=torch.bool)

def my_model_function():
    model = MyModel()
    # Initialize p0 with random values to avoid uninitialized parameters
    torch.nn.init.uniform_(model.p0)
    return model

def GetInput():
    return torch.rand(1, 1, 1, 26, 1, dtype=torch.float16)

