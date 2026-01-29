# torch.rand(10, 2, dtype=torch.float32)  # Input shape: B=10, C=2
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.tensor([1.0, 16.0], dtype=torch.float32))
        self.reduction = 'mean'
        self.ignore_index = -1
        self.label_smoothing = 0.1

    def forward(self, x):
        inputs, targets = x
        return F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction
        )

def my_model_function():
    return MyModel()

def GetInput():
    inputs = torch.rand(10, 2, dtype=torch.float32)
    targets = torch.randint(low=-1, high=2, size=(10,), dtype=torch.long)
    return (inputs, targets)

