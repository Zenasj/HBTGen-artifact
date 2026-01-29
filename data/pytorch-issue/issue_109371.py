# torch.rand(B, 256, dtype=torch.float32)
import torch
from torch import nn
from torch.nn import functional as F

class MyModel(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=32, num_layers=2, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.has_discrepancy = False  # Track layer(x) discrepancies

    def forward(self, x):
        self.has_discrepancy = False
        for i, layer in enumerate(self.layers):
            # Compare two identical layer calls in the same iteration
            out1 = layer(x)
            out2 = layer(x)
            if not torch.allclose(out1[0,0], out2[0,0], atol=1e-6):
                self.has_discrepancy = True
            # Proceed with third call for actual computation (matches original code flow)
            layer_out = layer(x)
            x = F.relu(layer_out) if i < self.num_layers - 1 else layer_out
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

def my_model_function():
    return MyModel(input_dim=256, hidden_dim=256, output_dim=32, num_layers=2, sigmoid_output=False)

def GetInput():
    return torch.rand(1, 256, dtype=torch.float32)

