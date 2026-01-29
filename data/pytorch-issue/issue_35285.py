# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Input shape inferred from the issue's example (1,3,32,32)
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 24, 3)

    def forward(self, x):
        return self.conv(x)

class Net_slice(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 24, 3)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :12], x[:, 12:24]

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Net()
        self.slice_net = Net_slice()

    def forward(self, x):
        full_output = self.net(x)
        slice1, slice2 = self.slice_net(x)
        expected_slice1 = full_output[:, :12]
        expected_slice2 = full_output[:, 12:24]
        # Return 1.0 if slices match within tolerance, else 0.0
        return torch.tensor(
            1.0 if (
                torch.allclose(slice1, expected_slice1, atol=1e-6) and
                torch.allclose(slice2, expected_slice2, atol=1e-6)
            ) else 0.0,
            dtype=torch.float32
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

