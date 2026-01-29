# torch.rand(B, 20, dtype=torch.float32)  # Inferred from optimizer's parameter shape [torch.rand(10,20)] and DataLoader batch_size=16
import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple linear layer to match parameter dimensions from optimizer setup
        self.fc = nn.Linear(20, 10)  # Input features=20 (from [torch.rand(10,20)] in optimizer)

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    return MyModel()  # No special initialization needed beyond default parameters

def GetInput():
    # Matches model input (batch, features) and DataLoader batch_size=16
    return torch.rand(16, 20, dtype=torch.float32)  # 16 is the batch size from the examples

