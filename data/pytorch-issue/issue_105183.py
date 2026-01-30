import torch.nn as nn

import torch
import onnxruntime

def func(x):
    return torch.nn.functional.avg_pool2d(x, 4)

export_output = torch.onnx.dynamo_export(func, torch.randn(1, 64, 32, 32))

onnxruntime.InferenceSession(export_output.model_proto.SerializeToString())