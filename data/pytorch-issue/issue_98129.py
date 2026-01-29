# torch.rand(1, 3, 16, 16, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ConvTranspose2d with output_padding=1 to trigger shape discrepancy
        self.conv = nn.ConvTranspose2d(
            in_channels=3,
            out_channels=6,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            dilation=1,
        )

    def forward(self, x):
        out = self.conv(x)
        actual_h, actual_w = out.size(2), out.size(3)
        
        # Extract parameters from the conv layer
        stride = self.conv.stride
        padding = self.conv.padding
        kernel_size = self.conv.kernel_size
        dilation = self.conv.dilation
        output_padding = self.conv.output_padding
        in_h, in_w = x.size(2), x.size(3)
        
        # Calculate expected output size using correct (with output_padding) and incorrect (JIT's) formulas
        correct_h = (in_h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
        correct_w = (in_w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
        
        # Incorrect formula (missing output_padding term)
        incorrect_h = (in_h - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + 1
        incorrect_w = (in_w - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + 1
        
        # Check if actual matches correct vs incorrect
        correct_match = (actual_h == correct_h) and (actual_w == correct_w)
        incorrect_match = (actual_h == incorrect_h) and (actual_w == incorrect_w)
        
        # Return 0.0 if actual matches correct formula, 1.0 otherwise
        return torch.tensor(0.0 if correct_match else 1.0)

def my_model_function():
    return MyModel()

def GetInput():
    # Input shape (B, C, H, W) matching the conv's requirements
    return torch.rand(1, 3, 16, 16, dtype=torch.float32)

