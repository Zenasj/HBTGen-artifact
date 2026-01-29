# torch.rand(1, 548, dtype=torch.float32)  # Inferred input shape from C++ code and logs

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 548  # Derived from C++ input tensor shape {1, 548}
        hidden_sizes = [100, 300, 200, 5]  # From user's 'layers' variable in issue
        self.neural_network = nn.ModuleList()
        sizes = [input_size] + hidden_sizes + [1]  # Final output size is 1

        for i in range(1, len(sizes)):
            self.neural_network.append(nn.Linear(sizes[i-1], sizes[i]))
            if i < len(sizes) - 1:  # Apply ReLU except after final layer
                self.neural_network.append(nn.ReLU())

    def forward(self, x):
        for layer in self.neural_network:
            x = layer(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input matching the expected [1, 548] shape
    return torch.rand(1, 548, dtype=torch.float32)

