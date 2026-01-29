# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLayer(nn.Module):
    def __init__(self, size):
        super(CustomLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(size), requires_grad=True)

    def forward(self, x):
        return x * self.weights

class MyModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super(MyModel, self).__init__()
        self.custom_layer = CustomLayer(input_size)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.custom_layer(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x) * 2
        print("woowoo")  # Preserved from original code example
        x = x * x
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 10)  # Matches input_size=10 in MyModel

