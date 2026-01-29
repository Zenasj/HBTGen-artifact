# torch.rand(16, 16, dtype=torch.float32) for each of A, B, C
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        A, B, C = inputs
        D = A
        for _ in range(5):
            D = torch.matmul(D, B)
        E = D + C
        for _ in range(5):
            E = E * torch.sigmoid(E)
        return E

def my_model_function():
    return MyModel()

def GetInput():
    A = torch.rand(16, 16, dtype=torch.float32)
    B = torch.rand(16, 16, dtype=torch.float32)
    C = torch.rand(16, 16, dtype=torch.float32)
    return (A, B, C)

