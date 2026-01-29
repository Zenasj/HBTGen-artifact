# torch.rand(B, C, H, W, dtype=torch.float32)  # Assuming input shape (1, 3, 64, 64)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Core operations sequence from issue description
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.sub_value = nn.Parameter(torch.tensor(0.5))  # Placeholder for sub operation
        self.div_value = nn.Parameter(torch.tensor(2.0))  # Placeholder for div operation
        # Example module representing the shared forward() call
        self.module_part = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
    
    def forward(self, x):
        # Replicate the operator sequence described in the issue
        x = x.to(x.device)  # Explicit device placement
        x = self.upsample(x)
        x = x.sub(self.sub_value)
        x = x.div(self.div_value)
        return self.module_part(x)

def my_model_function():
    # Returns the fused model with all described operations
    return MyModel()

def GetInput():
    # Generate random input matching expected shape (B=1, C=3, H=64, W=64)
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

