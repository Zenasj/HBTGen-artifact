# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 224 * 224, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # Dynamic indexing to trigger FX graph indexing operations
        idx = torch.arange(x.size(1), device=x.device)
        x = x.index_select(1, idx)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

