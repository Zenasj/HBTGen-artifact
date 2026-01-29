# torch.rand(3, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=3):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.ones(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.ones(ensemble_size, in_features, out_features))

    def forward(self, x):
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias
        return x

def my_model_function():
    return MyModel(in_features=3, out_features=3, ensemble_size=3)

def GetInput():
    return torch.rand(3, 3, dtype=torch.float32)

