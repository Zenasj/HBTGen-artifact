# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape for AvgPool2d test case
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # AvgPool2d layer matching the failing test case (test_avg_pool2d8_cuda)
        self.pool = nn.AvgPool2d(kernel_size=8, stride=8)  # '8' in test name implies kernel size 8

    def forward(self, x):
        return self.pool(x)

def my_model_function():
    # Returns model instance with default AvgPool2d configuration
    return MyModel()

def GetInput():
    # Input dimensions inferred from common AvgPool2d test patterns (B, C, H, W)
    # Using 8x8 kernel: input size must be divisible by 8 for exact pooling
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)  # 256 divisible by 8, matching kernel size

