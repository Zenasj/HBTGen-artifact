import torch
import torch.nn as nn

i = 1
in_shapes = [(1, 30, 25), (1, 15, 13), (1, 8, 7), (1, 4, 4), (1, 2, 2)]
in_shape = in_shapes[i]
out_shape = (1, 240, 200)
tensor = torch.zeros(in_shape)
up = 8 * 2 ** i
pad = up // 2 - 1
deconv = nn.ConvTranspose2d(1, 1, kernel_size=up, stride=up, padding=pad, bias=False)
assert deconv(tensor, output_size=out_shape).shape == out_shape