# torch.rand(B, 512, 8, 8, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, channels=512):
        super(MyModel, self).__init__()
        self.CHANNELS = channels  # Matches the original class's attribute naming

    def forward(self, x):
        # Calculate channel-wise mean and max pools then concatenate
        mean_pool = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        mean_pool = mean_pool.view(-1, self.CHANNELS)  # (B, C)
        
        max_pool = F.adaptive_max_pool2d(x, 1)  # (B, C, 1, 1)
        max_pool = max_pool.view(-1, self.CHANNELS)  # (B, C)
        
        return torch.cat((mean_pool, max_pool), dim=1)  # (B, 2C)

def my_model_function():
    # Returns model instance with default 512 channels (inferred from error traces)
    return MyModel()

def GetInput():
    # Generates input matching the model's expected dimensions (B, C, H, W)
    B, C, H, W = 2, 512, 8, 8  # Batch size 2, 512 channels, 8x8 spatial dimensions
    return torch.rand(B, C, H, W, dtype=torch.float32)

