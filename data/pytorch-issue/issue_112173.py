# torch.rand(2, 3, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1408)
        self.to("cuda")

    def forward(self, image_features):
        reward = self.linear(image_features)
        # Using non-inplace division to avoid memory leak
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        reward = (image_features[0] * image_features[1:]).sum(-1)
        return reward

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(2, 3).to("cuda")

