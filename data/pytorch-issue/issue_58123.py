import torch.nn as nn

import torch
import urllib
from PIL import Image
from torchvision import transforms

model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

model = model.cuda().float()
input_batch = input_batch.cuda().float()

with torch.no_grad():
    output = model(input_batch)
print("FP32 output:", output)

model = model.cuda().half()
input_batch = input_batch.cuda().half()

with torch.no_grad():
    output = model(input_batch)
print("FP16 output:", output)

import torch

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
input_tensor = torch.rand((1,3,512,512))

model = model.cuda().half()
input_tensor = input_tensor.cuda().half()
with torch.no_grad():
    output = model(input_tensor)

print("FP16 output:", output)

model = model.cuda().float()
input_tensor = input_tensor.cuda().float()
with torch.no_grad():
    output = model(input_tensor)

print("FP32 output:", output)

input_tensor = torch.rand((1,32,64,64)).half().cuda()
conv = torch.nn.Conv2d(32, 64, kernel_size=(3, 3)).half().cuda()
conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3)).half().cuda()
conv3 = torch.nn.Conv2d(128, 256, kernel_size=(3, 3)).half().cuda()
x = conv(input_tensor)
x = conv2(x)
x = conv3(x)
print(x)