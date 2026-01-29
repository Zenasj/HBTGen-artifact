# Inputs: (torch.rand(2,5,5, dtype=torch.float64), torch.randint(-1024,4, (2,5), dtype=torch.int32), torch.rand(5,5, dtype=torch.float64))
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, inputs):
        arg1, arg2, arg3 = inputs
        return torch.linalg.ldl_solve(arg1, arg2, arg3, hermitian=False)

def my_model_function():
    return MyModel()

def GetInput():
    arg_1 = torch.rand(2, 5, 5, dtype=torch.float64)
    arg_2 = torch.randint(-1024, 4, (2, 5), dtype=torch.int32)
    arg_3 = torch.rand(5, 5, dtype=torch.float64)
    return (arg_1, arg_2, arg_3)

