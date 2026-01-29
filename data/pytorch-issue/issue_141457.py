# torch.rand(B, 4, 100, dtype=torch.float32)  # Inferred input shape: (B, 4, 100)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the model structure
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # Apply softmax to ensure the last dimension sums to 1
        x = self.softmax(x)
        # Create a Categorical distribution
        dist = torch.distributions.Categorical(probs=x)
        # Sample from the distribution
        samples = dist.sample()
        return samples

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    B = 1  # Batch size
    C = 4  # Number of categories
    H = 100  # Length of each category
    input_tensor = torch.rand(B, C, H, dtype=torch.float32)
    return input_tensor

