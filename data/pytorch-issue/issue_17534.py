from __future__ import print_function
import torch

class SimpleModel1(torch.jit.ScriptModule):
    def __init__(self):
        super(SimpleModel1, self).__init__()

    @torch.jit.script_method
    def forward(self, dim : int):
        x = torch.ones([dim, 2], dtype=torch.float32)
        v = torch.ones(2, 1, dtype=torch.float32)
        v = x * v
        return x, v

model = SimpleModel1()

model_onnx = torch.onnx._export(model, torch.tensor(5), "simple1.onnx",
        verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        example_outputs=(torch.zeros(2,2), torch.zeros(2,1)))

class SimpleModel2(torch.jit.ScriptModule):
    def __init__(self):
        super(SimpleModel2, self).__init__()

    @torch.jit.script_method
    def forward(self, dim : int):
        x = torch.ones([2, 2], dtype=torch.float32)
        v = torch.ones(2, 1, dtype=torch.float32)
        v = x * v
        return x, v

model = SimpleModel2()
print(model.graph)
model_onnx = torch.onnx._export(model, torch.tensor(5), "simple2.onnx",
        verbose=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        example_outputs=(torch.zeros(2,2), torch.zeros(2,1)))