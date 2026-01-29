# torch.rand(1, 3, 8, 256, 256, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_size = (16, 320, 320)  # from benchmark input-output pairs
        self.mode = "nearest"
        self.align_corners = None  # "nearest" mode doesn't use align_corners

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, 
            size=self.output_size, 
            mode=self.mode, 
            align_corners=self.align_corners
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Matches the input shape from benchmark examples (channels_last format used in PR)
    x = torch.rand(1, 3, 8, 256, 256, dtype=torch.float32)
    return x.contiguous(memory_format=torch.channels_last_3d)

