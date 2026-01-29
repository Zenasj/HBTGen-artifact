# torch.rand(2, dtype=torch.float64)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.lower = -2.9  # Fixed from issue's example
        self.upper = -2.7  # Fixed from issue's example
        self.training = True  # Fixed from issue's example

    def forward(self, x):
        return F.rrelu(x, lower=self.lower, upper=self.upper, training=self.training)

def my_model_function():
    return MyModel()

def GetInput():
    # Reproduces exact input from the issue's test case, moved to CUDA
    input_tensor = torch.tensor([0.1250, 0.4313], dtype=torch.float64)
    input = input_tensor.detach().to('cuda').requires_grad_()
    return input

