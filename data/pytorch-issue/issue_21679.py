# torch.rand(1, 2, 2, 2, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm2d(2)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.fill_(3)
        self.bn.running_mean.data.fill_(2)
        self.bn.running_var.data.fill_(4)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    model = MyModel()
    model.eval()  # Set the model to evaluation mode
    return model

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.ones(1, 2, 2, 2, dtype=torch.float32)

