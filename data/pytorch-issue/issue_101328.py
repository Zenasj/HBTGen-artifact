import torch
import torch.nn as nn
import torch._dynamo as dynamo


mod = torch.jit.load('scriptmodule.pt')

input0 = torch.randn([1, 128, 48, 80], device="cuda")
input1 = torch.randn([1, 256, 24, 40], device="cuda")
input2 = torch.randn([1, 512, 12, 20], device="cuda")

# this runs fine
mod(input0, input1, input2)

# this will fail
torch.onnx.export(
    mod,
    [input0, input1, input2],
    "test.onnx",
    input_names=['input_0', 'input_1', 'input_2'],
    output_names=['output_0'],
    opset_version=14,
)