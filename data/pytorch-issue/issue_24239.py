# torch.rand(1, 3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        tensors = []
        for _ in range(3):
            random_tensor = torch.randn(1, 3, 3)
            mask = random_tensor > 0
            a = torch.randn(2)
            selected = a[mask.long()]
            tensors.append(selected)
        return torch.stack(tensors)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 3, dtype=torch.float32)

