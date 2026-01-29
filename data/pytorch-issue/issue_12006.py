# torch.rand(B, C, dtype=torch.float32)  # Inferred input shape (B, C) where B is batch size and C is number of features

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_features=256):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        return self.bn(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput(batch_size=5 * 20000, num_features=256):
    # Return a random tensor input that matches the input expected by MyModel
    input_tensor = torch.randn((batch_size, num_features), dtype=torch.float32, device='cuda')
    input_tensor.requires_grad = True
    return input_tensor

