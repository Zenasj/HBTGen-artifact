import torch.nn as nn

import torch
import onnxscript
import onnx

class GeluModel(torch.nn.Module):
    def forward(self, input_x):
        return torch.ops.aten.gelu(input_x)

microsoft_op = onnxscript.values.Opset(domain="com.microsoft", version=1)
from onnxscript import FLOAT

@onnxscript.script(microsoft_op)
def custom_aten_gelu(self: FLOAT, approximate: str = "none") -> FLOAT:
    return microsoft_op.Gelu(self)

x = torch.tensor([1.0])

onnx_program = torch.onnx.export(
    GeluModel().eval(),
    (x,),
    dynamo=True,
    custom_translation_table={
        torch.ops.aten.gelu.default: custom_aten_gelu,
    },
)

onnx_program.optimize()
print(onnx_program.model)
onnx_file_path="ms.onnx"
print("==============")
onnx_program.save(onnx_file_path)
onnx_model = onnx.load(onnx_file_path)
print(onnx.helper.printable_graph(onnx_model.graph))