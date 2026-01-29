# torch.rand(10, 10, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(10, 10)
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

    @property
    def encoder_layers(self):
        return [self.layer0, self.layer1, self.layer2]

    def forward(self, x):
        for layer in self.encoder_layers:
            output = layer(x)
            if layer == self.layer0:
                output = F.relu6(output)
            else:
                output = F.relu(output)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(10, 10)

