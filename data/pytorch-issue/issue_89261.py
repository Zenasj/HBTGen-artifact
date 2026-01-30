import torchvision

import torch
from torchvision.models import resnet18

model = resnet18()
dummy_input = torch.randn(1, 3, 224, 224)
print(torch.onnx.utils.unconvertible_ops(model, dummy_input)[1])