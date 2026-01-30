import torch.nn as nn

import torch

layer = torch.nn.PixelUnshuffle(2)

x = torch.rand(1, 1, 2, 2)

assert (layer(x) == x.view(1, 4, 1, 1)).all().item()  # passes

for opset_version in range(9, 14):
    try:
        torch.onnx.export(layer, x, "./pixel_unshuffle.onnx", opset_version=opset_version)
    except RuntimeError as err:
        print(err)