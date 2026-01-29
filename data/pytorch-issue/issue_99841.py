import torch
import torch.nn as nn

# torch.rand(1, 3, 128, 128, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_transpose2 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv_transpose_output = self.conv_transpose1(x)
        clamp_min_output = torch.clamp_min(self.conv_transpose2(conv_transpose_output), 3)
        clamp_max_output = torch.clamp_max(clamp_min_output, 0)
        div_output = torch.div(clamp_max_output, 6)
        return div_output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 128, 128)

