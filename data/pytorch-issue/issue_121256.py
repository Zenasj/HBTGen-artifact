import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float16) → Input shape is [1, 128, 128, 320]
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1x1 convolution layer causing the error (matches error logs: [16, 320, 1, 1])
        self.conv = nn.Conv2d(320, 16, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # Permute to channels-first format (as seen in error logs' PermuteView)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] → [B, C, H, W]
        return self.conv(x)

def my_model_function():
    # Initialize with float16 (matches error logs' tensor dtypes)
    model = MyModel().to(torch.float16).cuda()
    return model

def GetInput():
    # Shape from error logs' PermuteView data (size [1, 128, 128, 320])
    return torch.rand(1, 128, 128, 320, dtype=torch.float16, device="cuda")

