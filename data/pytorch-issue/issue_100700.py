import torch.nn as nn

import torch
import onnx
a = torch.tensor([10, 20, 30, 80], dtype=torch.int32)
def test():
    class SumInt32(torch.nn.Module):
        def forward(self, a):
            return torch.sum(a, dtype=torch.int32)

    sumi = SumInt32().eval()
    assert sumi(a).dtype == torch.int32
    print("Torch model output type matches input type")

    torch.onnx.export(sumi, (a), "/tmp/sumi_int32.onnx", opset_version=12)
    model = onnx.load("/tmp/sumi_int32.onnx")

    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT32
    print("ONNX model output type matches input type")
test()

import onnx
import torch

a = torch.tensor([10, 20, 30, 80], dtype=torch.int64)


def test():
    class SumInt64(torch.nn.Module):
        def forward(self, a):
            return torch.sum(a, dtype=torch.int64)

    sumi = SumInt64().eval()
    assert sumi(a).dtype == torch.int64
    print("Torch model output type matches input type")
    torch.onnx.export(sumi, (a), "/tmp/sumi_int64.onnx", opset_version=12)
    model = onnx.load("/tmp/sumi_int64.onnx")
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.INT64
    print("ONNX model output type matches input type")


test()