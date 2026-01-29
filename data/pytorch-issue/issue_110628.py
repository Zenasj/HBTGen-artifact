# torch.rand(2, 4, dtype=torch.float32)  # Shape after transpose (non-contiguous)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(4, 1)  # Matches input's second dimension (4)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    dp_size = 4
    mp_size = 2
    # Creates a non-contiguous tensor to match the DeviceMesh use-case
    x = torch.rand(dp_size, mp_size, dtype=torch.float32)
    return x.t()  # Transpose creates a non-contiguous view (2,4)

