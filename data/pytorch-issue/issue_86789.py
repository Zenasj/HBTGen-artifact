from io import BytesIO
import torch
import torch.onnx

def fn(x, y):
    a = torch.add(x, y)
    b = torch.mul(y, y)
    return a + b

inputs = [torch.ones(1), torch.ones(0)]
scripted = torch.jit.trace(fn, inputs)
onnx_bytes = BytesIO()
torch.onnx.export(scripted,
                    inputs,
                    onnx_bytes,
                    opset_version=14,
                    input_names = ["x", "y"],
                    output_names = ["output"])