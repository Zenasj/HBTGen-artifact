# torch.rand(B, 128, dtype=torch.float32).cuda()  # Assuming batch size B and 128 features
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Align with NCCL/CUDA context from the issue
    return model

def GetInput():
    # Matches the assumed input shape (B=32, features=128)
    return torch.rand(32, 128, device='cuda', dtype=torch.float32)

