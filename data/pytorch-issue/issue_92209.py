import torch.nn as nn

import torch


class GridSample(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, grid_tensor):

        output_tensor = torch.nn.functional.grid_sample(input=input_tensor,
                                                        grid=grid_tensor,
                                                        mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=False)

        return output_tensor


N = 4
C = 32
D_in = 64
H_in = 64
W_in = 64
D_out = 128
H_out = 128
W_out = 128

input_shape = [N, C, D_in, H_in, W_in]
grid_shape = [N, D_out, H_out, W_out, 3]
output_shape = [N, C, D_out, H_out, W_out]

input_tensor = torch.rand(*input_shape)
grid_tensor = torch.rand(*grid_shape)

grid_sample_module = GridSample()

output_tensor = grid_sample_module(input_tensor=input_tensor, grid_tensor=grid_tensor)

assert list(output_tensor.shape) == output_shape

torch.onnx.export(grid_sample_module,
                  {"input_tensor": input_tensor, "grid_tensor": grid_tensor},
                  "grid_sample.onnx",
                  verbose=False,
                  opset_version=16)