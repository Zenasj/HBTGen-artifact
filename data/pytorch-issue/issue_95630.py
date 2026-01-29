# torch.rand(B, C, H, W, dtype=...)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, x):
        output = self.forward(x)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 10
    input_features = 32
    return torch.rand(batch_size, input_features)

