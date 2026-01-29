# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.distributed as dist

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(16 * 32 * 32, 1)  # Output a single scalar for demonstration

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        x = self.fc(x)
        loss = x.sum()  # Dummy loss calculation
        
        # Ensure all-reduce is called by all processes when distributed is initialized
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        
        return loss

def my_model_function():
    model = MyModel()
    return model

def GetInput():
    # Random input tensor matching the expected shape (B, C, H, W)
    return torch.rand(4, 3, 32, 32, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")

