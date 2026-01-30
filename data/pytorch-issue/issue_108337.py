import torch.nn as nn

import torch
import torchvision

model = torchvision.models.detection.retinanet_resnet50_fpn(
        weights=torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT)
model = torch.compile(model, mode="default")

torch.save(model.state_dict(), "retina.pt")
print("model.state_dict:",model.state_dict().keys())

x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
model.eval()
exported_model = torch._dynamo.export(model, x)
torch.save(exported_model, "retina_retina.pt")

import torch

class MyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, x):
        zero = torch.zeros_like(x)
        scale = torch.max(zero, x)
        scale_factor = scale.item()
        return scale_factor

model = MyModel()

model = torch.compile(model, mode="default")

x = torch.rand(1)
scale = model(x)
print("result:", scale)

exported_model = torch._dynamo.export(model, x)
torch.save(exported_model, "test_net.pt")