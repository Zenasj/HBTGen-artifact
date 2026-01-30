import torch

x = torch.ones(1, 3, 224, 224, requires_grad=True)
torch.onnx.export(model, x, "faster.onnx", export_params=True)

x = torch.ones(1, 3, 224, 224, requires_grad=True)
torch.onnx.export(model, x, "faster.onnx", export_params=True)