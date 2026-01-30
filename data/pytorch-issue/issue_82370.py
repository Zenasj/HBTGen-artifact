import torch
import torchvision

x = torch.randn(1, 3, 299, 299)
model = torchvision.models.inception_v3()
model = torch.jit.script(model)
torch.onnx.export(model, x, 'model.onnx', verbose=True)