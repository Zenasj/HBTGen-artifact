import torch

model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)

model = torch.hub.load('rwightman/pytorch-pretrained-gluonresnet', 'gluon_resnet34_v1b', pretrained=True)