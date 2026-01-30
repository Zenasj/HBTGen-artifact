import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torch.nn import functional as F
import torch

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            print(layer(x)[0,0])
            print(layer(x)[0,0])
            print(sum(layer.weight[0]*x[0])+layer.bias[0])
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            continue
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


if __name__ == '__main__':
    mlp = MLP(256,256,32, 2).cuda()
    x = torch.ones(1,256).cuda()
    results = mlp(x)