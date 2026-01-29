# torch.rand(2, 3, 3, 3, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dropout = nn.Dropout(p=0.5)  # Core module causing the bug

    def forward(self, x):
        return self.dropout(x)

def my_model_function():
    return MyModel()  # Returns the faulty Dropout model

def GetInput():
    # Reproduces discontiguous channels-last input as in the issue
    base = torch.empty(2, 3, 3, 6, device="cuda", memory_format=torch.channels_last)
    base.normal_(0, 1)  # Random initialization instead of fixed arange
    input_tensor = base[..., ::2]  # Creates discontiguous view
    return input_tensor

