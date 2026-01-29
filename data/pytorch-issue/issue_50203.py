# torch.rand(4), torch.rand(3, 3), torch.rand(3, 4)  # Input shapes (tuple of tensors)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        input_tensor, mat1, mat2 = inputs
        return torch.addmm(input_tensor, mat1, mat2)  # Correct 3-argument call

def my_model_function():
    return MyModel()

def GetInput():
    input = torch.rand(4)          # 1D tensor (matches first argument)
    mat1 = torch.rand(3, 3)        # 3x3 matrix
    mat2 = torch.rand(3, 4)        # 3x4 matrix
    return (input, mat1, mat2)     # Tuple of three tensors required by forward()

