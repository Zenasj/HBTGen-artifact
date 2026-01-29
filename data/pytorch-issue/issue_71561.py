# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Example input shape and dtype
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(32 * 32 * 3, 64)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Example input for a simple neural network (batch, channels, height, width)
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

