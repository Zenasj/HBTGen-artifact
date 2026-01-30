import torch

torch.onnx._export(loaded_m, x, 'model.onnx', opset_version=11, example_outputs=torch.rand((1, 1000)))