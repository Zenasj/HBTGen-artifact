# torch.rand(7, 5, dtype=torch.float32)  # Inferred input shape from the issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(5, 4, bias=False)
        self.linear2 = nn.Linear(4, 5, bias=False)
        self.linear1.register_full_backward_hook(lambda m, gin, gout: None)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(7, 5, dtype=torch.float32).cuda()

