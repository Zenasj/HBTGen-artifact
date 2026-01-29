# torch.rand(B, C, L, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_fps=50, output_fps=30):
        super().__init__()
        self.input_fps = input_fps
        self.output_fps = output_fps

    def forward(self, features):
        # Swap channels and length dimensions to apply interpolation on length
        features = features.transpose(1, 2)
        # Compute target length based on input/output FPS ratio
        seq_len = features.shape[2] / self.input_fps
        output_len = int(seq_len * self.output_fps) + 1
        # Use ATen's upsample_linear1d to handle dynamic shapes
        output_features = torch.ops.aten.upsample_linear1d(
            features, output_size=[output_len], align_corners=True
        )
        # Transpose back to original channel dimension order
        return output_features.transpose(1, 2)

def my_model_function():
    # Initialize with default FPS values from the original code
    return MyModel(input_fps=50, output_fps=30)

def GetInput():
    # Based on first entry in sizesTen: [1, 336, 512]
    return torch.rand(1, 336, 512, dtype=torch.float32)

