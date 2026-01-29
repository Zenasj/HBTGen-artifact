# torch.rand(3, dtype=torch.float64, device='cuda'), torch.rand(1,2,3,4,5, dtype=torch.float32, device='cuda'), torch.empty(4, dtype=torch.float32, device='cuda')

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Scenario1()
        self.model2 = Scenario2()
        self.model3 = Scenario3()
    
    def forward(self, inputs):
        input1, input2, input3 = inputs
        out1 = self.model1(input1)
        out2 = self.model2(input2)
        out3 = self.model3(input3)
        return (out1, out2, out3)

class Scenario1(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = torch.tensor([1, 2, 3], dtype=torch.int64, device='cuda')
    
    def forward(self, input):
        out = self.out
        out = F.tanhshrink(out)
        return torch.logical_not(input, out=out)

class Scenario2(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        numel_val = torch.numel(input)
        tensor_val = torch.tensor(-3, dtype=torch.float32, device='cuda')
        res = torch.div(numel_val, tensor_val)
        res = torch.mul(res, torch.tensor(-6, dtype=torch.float32, device='cuda'))
        return res

class Scenario3(nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.randint(0, 2, (4, 1), dtype=torch.bool, device='cuda')
        self.alpha = 10
    
    def forward(self, input):
        other = self.other
        alpha = self.alpha
        other = torch.cos(other)
        other = torch.mul(other, torch.tensor(-12, dtype=torch.float32, device='cuda'))
        input = F.relu(input)
        alpha = torch.sub(alpha, torch.tensor(-13, dtype=torch.float32, device='cuda'))
        res = input.add(other, alpha=alpha)
        res = torch.sin(res)
        return res

def my_model_function():
    return MyModel()

def GetInput():
    input1 = torch.rand(3, dtype=torch.float64, device='cuda')
    input2 = torch.rand(1, 2, 3, 4, 5, dtype=torch.float32, device='cuda')
    input3 = torch.empty(4, dtype=torch.float32, memory_format=torch.contiguous_format, device='cuda')
    input3.uniform_(0, 31)
    return (input1, input2, input3)

