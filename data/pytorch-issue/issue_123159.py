import timm
import torch

model = timm.create_model('resnet50').cuda()
model = torch.compile(model)
model(torch.randn((8,3,128,128)).cuda())