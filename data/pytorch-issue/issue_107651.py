# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.distributed as dist
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simulate distributed setup that triggers the NCCL error
        try:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:10001',
                                   rank=0, world_size=1)
        except ValueError as e:
            # Re-raise as DistBackendError per PR's new error hierarchy
            raise dist.DistBackendError("NCCL backend not available") from e
        # Example model structure (Moco-like)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    
    def forward(self, x):
        return self.encoder(x).flatten(1)

def my_model_function():
    # Returns an instance with mocked distributed setup
    return MyModel()

def GetInput():
    # Random input tensor matching expected dimensions
    B = 2  # batch size
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

