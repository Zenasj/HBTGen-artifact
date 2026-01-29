# torch.rand(B, 1, 176, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.gap(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = torch.log_softmax(x, dim=1)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 176)  # Example input with batch_size=1

