import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, H * W)

model = Model()

input_tensor = torch.rand((2, 64, 128, 128))
torch.onnx.export(
    model,
    (input_tensor,),
    "model.onnx",
    input_names = ["input"],
    output_names = ["output"],
    dynamo = True,
    dynamic_axes = { "input": {0: "batch", 2: "height", 3: "width"} }
)