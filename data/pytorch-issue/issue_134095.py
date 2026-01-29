# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)  # Matches original model initialization
        self.net1 = nn.Linear(8, 16)
        self.net2 = nn.Linear(16, 32)
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Linear(64, 8)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        return x

def my_model_function():
    # Returns initialized model on CUDA (matches original test setup)
    model = MyModel()
    model.cuda()
    return model

def GetInput():
    # Returns CUDA tensor matching the model's expected input
    return torch.rand(8, 8, device="cuda")

