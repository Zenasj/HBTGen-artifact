# torch.rand(CHANNELS, 16, 24, 12, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.W = nn.Parameter(torch.ones(18, requires_grad=True))
        self.b = nn.Parameter(torch.zeros(18, requires_grad=True))

    def forward(self, x):
        CHANNELS, _, _, _ = x.size()
        numel = x.size(1) * x.size(2) * x.size(3)
        x_s = x.sum(dim=(1, 2, 3))
        x_mean = x_s / numel
        x = (x - x_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        output = x * self.W.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + self.b.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return output

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    CHANNELS = 18
    input = torch.rand(CHANNELS, 16, 24, 12, dtype=torch.float32)
    input[:8] -= 0.5
    input[8:] += 0.5
    return input

