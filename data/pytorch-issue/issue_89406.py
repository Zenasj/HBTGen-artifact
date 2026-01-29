# torch.rand(B, C, H, W, dtype=...)  # Input shape: (batch_size, channels, height, width)
import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 100)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.mlp1 = Mlp()
        self.mlp2 = Mlp()

    def forward(self, x):
        return self.mlp2(self.mlp1(x))

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 32  # Example batch size
    input_tensor = torch.rand(batch_size, 100)  # Input shape: (batch_size, 100)
    return input_tensor

