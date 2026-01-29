import torch
import torch.nn as nn

# torch.rand(B, C, H, W, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # Problematic BatchNorm/InstanceNorm layers (track_running_stats=False, stats set to None)
        self.bn = nn.BatchNorm2d(64, track_running_stats=False)
        self.inorm = nn.InstanceNorm2d(64, track_running_stats=False)
        # Explicitly set running stats to None to replicate the issue
        self.bn.running_mean = None
        self.bn.running_var = None
        self.inorm.running_mean = None
        self.inorm.running_var = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.inorm(x)
        return x

def my_model_function():
    model = MyModel()
    # Initialize weights (optional, but ensures model is ready for inference)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return model

def enable_running_stats(model, device):
    """Temporarily enable track_running_stats and initialize running stats."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            num_features = module.num_features
            module.running_mean = torch.zeros(num_features, device=device)
            module.running_var = torch.ones(num_features, device=device)
            module.track_running_stats = True

def disable_running_stats(model):
    """Revert track_running_stats to original state (False)."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

def GetInput():
    return torch.rand(1, 3, 64, 64, dtype=torch.float32)

