import torch
import torchvision

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn().eval().cuda()

input1 = torch.rand((3, 300, 400)).cuda()
input2 = torch.rand((3, 500, 400)).cuda()

compiled = torch.compile(backend="eager")(model)

out = compiled([input1, input2])