# torch.rand(B, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.scale = 0.5  # Simulates scaling mentioned in FusedMatMul context

    def forward(self, x):
        x = self.fc(x)
        return x * self.scale  # Scaling operation fused with MatMul in ONNX

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10)  # Matches input shape expected by the model

