# torch.rand(B, 128, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(128)
        self.linear = nn.Linear(128, 4)
    
    def forward(self, x):
        # Bypass batch norm during training with batch size 1
        if self.training and x.size(0) == 1:
            pass  # Skip batch norm to avoid division-by-zero in mean/variance calculation
        else:
            x = self.bn(x)
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a batch of size 1 (problematic case) to validate the workaround
    return torch.rand(1, 128, dtype=torch.float32)

