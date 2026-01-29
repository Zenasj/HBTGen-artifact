# torch.rand(B, 2, dtype=torch.float32)  # Inferred input shape for XOR-like model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.activation(self.layer2(x))
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Batch size (arbitrary choice for testing)
    return torch.rand(B, 2, dtype=torch.float32)

