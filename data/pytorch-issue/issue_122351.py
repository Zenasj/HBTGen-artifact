import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        out = self.linear(x)
        return out

with torch.onnx.enable_fake_mode() as fake_context:
    x = torch.rand(5, 2, 2)
    model = Model()
    export_options = ExportOptions(fake_context=fake_context)
    onnx_program = torch.onnx.dynamo_export(
        model, x, export_options=export_options
    )

onnx_program.save("/path/to/model.onnx", state_dict="/path/to/checkpoint.pt")
onnx_program.save("/path/to/model.onnx", state_dict="/path/to/checkpoint.pt")