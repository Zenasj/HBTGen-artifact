import torch.nn as nn

import torch
class Test(torch.nn.Module):
    def forward(self, x):
        return torch.arange(end=x.shape[0], dtype=torch.float16)

test = Test()

torch.onnx.export(test, torch.randn(1), "test.onnx", dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}}, input_names=["input"], output_names=["output"])

import onnxruntime
sess = onnxruntime.InferenceSession("test.onnx")