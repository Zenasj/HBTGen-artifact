# torch.rand(1, 10, dtype=torch.float32)  # Inferred input shape for the model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 10, dtype=torch.float32)

# Linear learning rate decay function
def linear_lr_decay(epoch, max_epochs, lr_start, lr_end):
    slope = (lr_end - lr_start) / (max_epochs - 1)
    return (lr_start + slope * epoch) / lr_start

# Number of epochs, initial/final learning rates
num_epochs = 100
lr_start = 4e-4
lr_end = 1e-5

# Dummy optimizer (replace with your own optimizer)
model = my_model_function()
optimizer = optim.Adam(model.parameters(), lr=lr_start)

# Create a LambdaLR scheduler with linear decay
scheduler = LambdaLR(optimizer, lambda epoch: linear_lr_decay(epoch, num_epochs, lr_start, lr_end))

