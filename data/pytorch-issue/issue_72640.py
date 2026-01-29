# torch.rand(2, 1000, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1000, 2000)
    
    def forward(self, x):
        full_out = self.linear(x)
        first_out = self.linear(x[:1])  # Compute output for first element
        # Return absolute difference between first element in full batch vs isolated batch
        return torch.abs(full_out[0, 0] - first_out[0, 0])

def my_model_function():
    model = MyModel()
    model.eval()  # Matches original example's .eval() call
    return model

def GetInput():
    return torch.rand(2, 1000, dtype=torch.float32)

