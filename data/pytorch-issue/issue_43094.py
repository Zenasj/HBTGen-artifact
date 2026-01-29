import torch
import numpy as np
from torch import nn

# torch.rand(B, 3, 64, 128, 128, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        shape = (3, 64, 128, 128)
        input_size = int(np.prod(shape))  # Convert numpy int64 to Python int for JIT compatibility
        self.fc = nn.Linear(input_size, 10)
    
    def forward(self, x):
        # Flatten input tensor before applying linear layer
        return self.fc(x.view(x.size(0), -1))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 64, 128, 128, dtype=torch.float32)

