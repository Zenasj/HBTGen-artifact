# torch.rand(B, C, H, W, dtype=torch.float32)  # B=1, C=3, H=1024, W=1024
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", None)  # Track kernel as a buffer for device management

    def set_radius(self, radius: int):
        """Set kernel size based on radius to replicate the failing scenario."""
        kernel_size = 2 * radius + 1
        self.kernel = torch.zeros(
            [3, 1, 1, kernel_size],  # [out_channels, in_channels_per_group, kernel_h, kernel_w]
            dtype=torch.float32,
            device=self.kernel.device if self.kernel is not None else "cpu",
        )

    def forward(self, x):
        if self.kernel is None:
            raise RuntimeError("Kernel must be set via set_radius() before forward pass.")
        return F.conv2d(x, self.kernel, groups=3)

def my_model_function():
    """Returns an instance of MyModel initialized with radius=7 (crash trigger point)."""
    model = MyModel()
    model.set_radius(7)  # Radius 7 is where the crash occurs in the original issue
    return model

def GetInput():
    """Generates a permuted input tensor matching the expected model input shape."""
    # Original input shape before permutation: [B, H, W, C] → permuted to [B, C, H, W]
    return torch.rand([1, 3, 1024, 1024], dtype=torch.float32)

