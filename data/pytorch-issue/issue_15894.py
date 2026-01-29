# torch.rand(B, C, H, W, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Custom SyncBatchNorm layer (placeholder using PyTorch's native implementation)
        self.sync_bn = nn.SyncBatchNorm(64)  # Example: 64 channels
        # Note: Actual implementation would require the compiled '_ext.sync_bn_lib' extension

    def forward(self, x):
        return self.sync_bn(x)

def my_model_function():
    # Returns an instance using standard initialization (replace with custom weights if needed)
    return MyModel()

def GetInput():
    # Matches expected input shape (B, C, H, W) for SyncBatchNorm
    return torch.rand(4, 64, 64, 64, dtype=torch.float32, device='cuda')  # Example dimensions

