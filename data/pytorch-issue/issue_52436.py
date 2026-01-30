import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x):
        return torch.pow(2, x)

torch.onnx.export(
    Model(),
    (torch.randn(1, 10),),
    "model.onnx",
    input_names=["x"],
    output_names=["y"],
    verbose=True
)