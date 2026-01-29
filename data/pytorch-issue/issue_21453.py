# torch.rand(B, C, H, W, dtype=torch.float)  # Input shape: (1, 3, 299, 299)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, transform_input=True):
        super(MyModel, self).__init__()
        self.transform_input = transform_input
        # Constants with mixed dtypes to simulate ONNX type mismatch issue
        self.register_buffer('scale0', torch.tensor(0.229 / 0.5, dtype=torch.float))
        self.register_buffer('bias0', torch.tensor((0.485 - 0.5)/0.5, dtype=torch.double))
        self.register_buffer('scale1', torch.tensor(0.224 / 0.5, dtype=torch.float))
        self.register_buffer('bias1', torch.tensor((0.456 - 0.5)/0.5, dtype=torch.double))
        self.register_buffer('scale2', torch.tensor(0.225 / 0.5, dtype=torch.float))
        self.register_buffer('bias2', torch.tensor((0.406 - 0.5)/0.5, dtype=torch.double))

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * self.scale0 + self.bias0
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * self.scale1 + self.bias1
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * self.scale2 + self.bias2
            x = torch.cat((x_ch0.float(), x_ch1.float(), x_ch2.float()), 1)  # Force float for ONNX compatibility
        # Placeholder for remaining InceptionV3 layers (not provided in issue)
        return x  # Output after transform_input processing

def my_model_function():
    return MyModel(transform_input=True)

def GetInput():
    # Matches InceptionV3 input requirements (B, C, H, W)
    return torch.rand(1, 3, 299, 299, dtype=torch.float)

