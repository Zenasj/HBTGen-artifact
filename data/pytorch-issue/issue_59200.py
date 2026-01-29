# torch.rand(1, 1, dtype=torch.float) - 0.5  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    shapes = (1, 1)
    X = torch.rand(*shapes, dtype=torch.float) - 0.5
    min_val = torch.min(X)
    max_val = torch.max(X)
    X_zero_point = int(torch.randint(-128, 127, (1,)))
    num_bins = 2 ** 8
    X_scale = float(max_val - min_val) / num_bins
    qx = torch.quantize_per_tensor(X, X_scale, X_zero_point, torch.quint8)
    return qx.dequantize()  # Dequantize to match the input type expected by MyModel

