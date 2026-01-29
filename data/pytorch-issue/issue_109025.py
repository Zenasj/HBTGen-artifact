# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn
import functorch

class MyModel(nn.Module):
    def __init__(self, num_models=5):
        super().__init__()
        # Simulate stacked parameters for an ensemble of models
        self.weight = nn.Parameter(torch.randn(num_models, 5, 3))  # (num_models, out_features, in_features)
        self.bias = nn.Parameter(torch.randn(num_models, 5))       # (num_models, out_features)

    def forward(self, x):
        def fmodel_single(weight, bias, x):
            # Linear layer computation with transpose (triggers the error)
            return x @ weight.t() + bias  # Transpose of weight (5,3) becomes (3,5)

        # Vectorize over model parameters (first dimension)
        return functorch.vmap(fmodel_single, in_dims=(0, 0, None))(self.weight, self.bias, x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

