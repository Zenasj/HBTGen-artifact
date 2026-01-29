# torch.rand(B, 1, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class ZeroChannelsConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.out_channels == 0:
            self.weight = nn.Parameter(torch.empty(0))
            self.bias = None if self.bias is None else nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.out_channels == 0:
            # Compute output spatial dimensions using standard Conv2d formula
            kernel_size = self.kernel_size
            stride = self.stride
            padding = self.padding
            dilation = self.dilation
            H = x.shape[2]
            W = x.shape[3]
            H_out = (H + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - 1) // stride[0] + 1
            W_out = (W + 2*padding[1] - dilation[1]*(kernel_size[1]-1) - 1) // stride[1] + 1
            return torch.empty(x.size(0), 0, H_out, W_out, device=x.device, dtype=x.dtype)
        else:
            return super().forward(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ZeroChannelsConv2d(in_channels=1, out_channels=0, kernel_size=(2, 2))
        
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 4, 4)  # Example input with shape (B=1, C=1, H=4, W=4)

