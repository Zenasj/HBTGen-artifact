# torch.rand(1, 21, 1, 40, dtype=torch.float16), torch.rand(1, 1, 21, 18, 1, dtype=torch.float16)

import torch
import torch.nn as nn

class Model0Sub(nn.Module):
    def __init__(self, v0_0):
        super().__init__()
        self.v0_0 = v0_0  # Shared parameter from parent MyModel

    def forward(self, *args):
        getitem = args[0]
        getitem_1 = args[1]
        mul = torch.mul(self.v0_0, getitem)
        max_1 = torch.max(getitem_1, mul)
        max_2 = max_1.max(3)
        getattr_1 = max_2.values
        gt = torch.gt(mul, getattr_1)
        return (gt, )  # Model0's output format

class Model1Sub(nn.Module):
    def __init__(self, v0_0):
        super().__init__()
        self.v0_0 = v0_0  # Shared parameter from parent MyModel

    def forward(self, *args):
        getitem = args[0]
        getitem_1 = args[1]
        mul = torch.mul(self.v0_0, getitem)
        cat_1 = torch.cat((mul, mul), dim=0)
        max_1 = torch.max(getitem_1, mul)
        max_2 = max_1.max(3)
        getattr_1 = max_2.values
        gt = torch.gt(mul, getattr_1)
        return (cat_1, gt)  # Model1's output format (includes extra cat_1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize shared parameter (matches original p0)
        self.v0_0 = torch.tensor([5.6484], dtype=torch.float16, device='cuda')
        self.model0 = Model0Sub(self.v0_0)
        self.model1 = Model1Sub(self.v0_0)

    def forward(self, inputs):
        out0 = self.model0(*inputs)
        out1 = self.model1(*inputs)
        # Extract GT tensors from both models (output names: v7_0 in both)
        gt0 = out0[0]
        gt1 = out1[1]
        # Return comparison result (True if all elements match)
        return torch.all(gt0 == gt1)  # Returns 0-dim torch.bool tensor

def my_model_function():
    return MyModel()

def GetInput():
    # Match input shapes from original repro script
    input0 = torch.rand(1, 21, 1, 40, dtype=torch.float16, device='cuda')
    input1 = torch.rand(1, 1, 21, 18, 1, dtype=torch.float16, device='cuda')
    return (input0, input1)

