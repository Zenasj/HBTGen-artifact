# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (batch_size, channels, height, width)

import torch
import torch.nn as nn
import torch.distributed as dist

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.self_encoder_conv1 = nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)).cuda()
        self.self_encoder_conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)).cuda()

    def forward(self, x: torch.Tensor):
        x = self.self_encoder_conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.self_encoder_conv2(x)
        x = torch.nn.functional.relu(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    batch_size, channels, height, width = 32, 3, 256, 256
    return torch.rand(batch_size, channels, height, width, dtype=torch.float32, device='cuda')

