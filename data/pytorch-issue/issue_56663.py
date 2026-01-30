import torch.nn as nn

import torch


@torch.jit.script
def slice_helper(x, offset):
    return x[:, :, :offset]


class Model(torch.nn.Module):
    def forward(self, x):
        x = slice_helper(x, x.size(2))
        xt = x.transpose(1, 2)
        return xt


m = Model()
i = (torch.randn(1, 2, 3),)

torch.onnx.export(
    m,
    i,
    "model.onnx",
    input_names=["INPUT_0"],
    output_names=["OUTPUT_0"],
    dynamic_axes={"INPUT_0": {0: "batch_size"}, "OUTPUT_0": {0: "batch_size"}},
    opset_version=12,
    verbose=True
)