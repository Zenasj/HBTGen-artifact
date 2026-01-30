import torch
import torchvision
torch.onnx.export(model, z, "test_stylegan.onnx", verbose=True)