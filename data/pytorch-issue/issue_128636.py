# torch.rand(B, 50, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 80)
        self.fc2 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, 32)
        self.fc4 = nn.Linear(32, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        h1 = nn.functional.relu(self.fc1(x))
        h2 = nn.functional.relu(self.fc2(h1))
        h3 = nn.functional.relu(self.fc3(h2))
        out = self.fc4(h3)
        if target is not None:
            loss = self.loss_fn(out, target)
            return out.detach(), loss  # Critical fix to prevent grad mismatch
        return out

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 50, dtype=torch.float32)

