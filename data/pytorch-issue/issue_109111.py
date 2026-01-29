import math
import torch
import torch.nn as nn

# torch.rand(B, 10, dtype=torch.float)
class MLPResidualLayer(nn.Module):
    def __init__(self, dim):
        super(MLPResidualLayer, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return residual + x

class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.input_layer = nn.Linear(10, 512)
        self.resnet = MLPResidualLayer(512)
        self.layer_norm = nn.LayerNorm(512)
        self.output_heads = VectorizedLinear(512, 3, 6)
        self.num_heads = 6

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer_norm(self.resnet(x))
        x = x.unsqueeze(0).repeat(self.num_heads, 1, 1)
        vals = self.output_heads(x).transpose(0, 1)
        return vals

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(256, 10, dtype=torch.float)

