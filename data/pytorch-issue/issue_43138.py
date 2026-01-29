# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model here. For this issue, we will define a simple model that uses Categorical distribution.
        self.probs = None

    def forward(self, x):
        # Convert input to float and normalize
        self.probs = x.float() / x.sum(-1, keepdim=True)
        return self.probs

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    population_size = 1000
    max_score = 1000
    return torch.randint(max_score, size=(100 * population_size,))

