import torch

qmodel = xxxx # trace and quantize
tensor_x = torch.randn(1, 3, 224, 224)
onnx_model = dynamo_export(qmodel, tensor_x)  # failed!