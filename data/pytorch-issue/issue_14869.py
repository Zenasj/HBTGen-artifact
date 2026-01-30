import torch
print(torch.__version__)

import torchvision

m = torchvision.models.resnet50()
x = torch.rand((1, 3, 224, 224))
traced_m = torch.jit.trace(m, x)
f = 'model.pt'
torch.jit.save(traced_m, f)
loaded_m = torch.jit.load(f)
torch.onnx._export(loaded_m, x, 'model.onnx', example_outputs=loaded_m(x))