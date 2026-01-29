# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    # Initialize weights for reproducibility
    torch.manual_seed(0)
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.normal_(param, mean=0, std=0.1)
        else:
            nn.init.constant_(param, 0.01)
    return model

def GetInput():
    B = 1  # Batch size
    return torch.rand(B, 10, dtype=torch.float32)

