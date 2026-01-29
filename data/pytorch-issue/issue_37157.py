# torch.rand(6, 256, 50, 50, dtype=torch.half, device="cuda")  # Inferred input shape and dtype
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True
        )
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    # Replicate original setup: move to CUDA then cast to FP16
    return model.cuda().to(dtype=torch.half)

def GetInput():
    # Matches input shape and dtype from issue's reproduction code
    return torch.randn(6, 256, 50, 50, dtype=torch.half, device="cuda")

