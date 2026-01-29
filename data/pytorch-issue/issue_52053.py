import torch
import torch.nn as nn

# torch.rand(B, 10, dtype=torch.float)  # Inferred input shape for a linear layer
class MyModel(nn.Module):
    def __init__(self: "MyModel", in_features: int, out_features: int):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    return MyModel(in_features=10, out_features=5)

def GetInput():
    return torch.rand(3, 10)  # Batch size 3, 10 features

