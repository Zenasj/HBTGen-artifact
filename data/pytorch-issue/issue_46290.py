# torch.rand(2, 16, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters sampled from issue's test configuration:
        # in_channels=16, out_channels=16 (groups=in_channels), kernel_size=(3,3), stride=(2,2)
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=16,  # Matches in_channels for grouped convolution
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=0,
            output_padding=(1, 1),
            dilation=1,
            groups=16  # groups=in_channels (16)
        )
    
    def forward(self, x):
        return self.conv_transpose(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (B, C, H, W) from the issue's test parameters
    return torch.rand(2, 16, 32, 32, dtype=torch.float32)

