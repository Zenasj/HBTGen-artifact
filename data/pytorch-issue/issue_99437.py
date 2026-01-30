import torch

@dynamo.optimize("onnxrt"/"tvm", dynamic=True)
def func(a, b):
    x = a / (torch.abs(a) + 1)
    dynamo.graph_break()
    if b.sum() < 0:
        b = b * -1

func(torch.randn(10), torch.randn(10))