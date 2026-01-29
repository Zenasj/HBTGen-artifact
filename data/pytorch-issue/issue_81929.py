# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # net1 and bias are all unused params.
        self.net1 = nn.Linear(10, 5, bias=False)
        self.bias = nn.Parameter(torch.zeros(5))
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(x).sum()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Assuming the input shape is (B, 10) where B is the batch size
    B = 4  # Example batch size
    return torch.rand(B, 10, dtype=torch.float32)

