# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(16 * 224 * 224, 10)  # Example shape based on input assumptions

    def forward(self, x):
        # Use fill_, which is now supported on meta tensors (in-place op added in PR)
        x = x.fill_(1.0) if x.is_meta else x  # Meta tensor compatibility example
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Random input matching the assumed shape (B=2, C=3, H=224, W=224)
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

