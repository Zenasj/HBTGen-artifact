# torch.rand(25, dtype=torch.float32)
import torch
import torch.nn as nn

class Model0(nn.Module):
    def forward(self, *args):
        getitem = args[0]
        sin = torch.sin(getitem)
        gelu = torch._C._nn.gelu(getitem)
        sigmoid = torch.sigmoid(gelu)
        sub = torch.sub(gelu, gelu)
        return (sin, sigmoid, sub)

class Model1(nn.Module):
    def forward(self, *args):
        getitem = args[0]
        sin = torch.sin(getitem)
        gelu = torch._C._nn.gelu(getitem)
        sigmoid = torch.sigmoid(gelu)
        sub = torch.sub(gelu, gelu)
        return (sigmoid, sub, sin)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model0 = Model0()
        self.model1 = Model1()
        self.output_names_0 = ['v5_0', 'v4_0', 'v1_0']
        self.output_names_1 = ['v4_0', 'v1_0', 'v5_0']
        self.output_name_dict = {'v4_0': 'v4_0', 'v5_0': 'v5_0', 'v1_0': 'v1_0'}

    def forward(self, x):
        outputs0 = self.model0(x)
        outputs1 = self.model1(x)
        output_dict0 = dict(zip(self.output_names_0, outputs0))
        output_dict1 = dict(zip(self.output_names_1, outputs1))
        all_close = True
        for name in self.output_name_dict:
            a = output_dict0[name]
            b = output_dict1[name]
            if not torch.allclose(a, b, rtol=1, atol=0):
                all_close = False
                break
        return torch.tensor(all_close, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(25, dtype=torch.float32)

