import torch.nn as nn

import torch
class Model1(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 99))
        return x

x = torch.randn(32, 1, 500)
model_onnx = torch.onnx._export(Model1(), x, "tuple_value.onnx",
                                verbose=True,
                                operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                                example_outputs=Model1()(x))