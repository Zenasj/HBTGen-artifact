# torch.rand(1024, 1024, dtype=torch.float), torch.rand(1024, 1024, dtype=torch.float)

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        matrix1, matrix2 = inputs
        if matrix1.shape[1] != matrix2.shape[0]:
            raise ValueError("Matrices have incompatible shapes for multiplication.")
        if torch.cuda.is_available():
            matrix1 = matrix1.to('cuda')
            matrix2 = matrix2.to('cuda')
        return torch.matmul(matrix1, matrix2)

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.rand(1024, 1024, dtype=torch.float),
        torch.rand(1024, 1024, dtype=torch.float)
    )

