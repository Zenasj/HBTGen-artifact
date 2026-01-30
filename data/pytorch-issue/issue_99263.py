import torch
from torchvision import models

model = models.resnet152(pretrained=True)
for param in model.parameters():
	param.requires_grad = False

example_input = torch.rand(4, 3, 224, 224)
script_module = torch.jit.trace(model, example_input)
script_module.save('resnet152.pt')