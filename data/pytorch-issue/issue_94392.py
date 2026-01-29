# torch.rand(2, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        self.linear2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def train_step(self, x, optimizer):
        loss = self(x).mean()
        loss.backward()
        optimizer.step()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 4, dtype=torch.float32)

