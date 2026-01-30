import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)

model = torchvision.models.quantization.resnet.resnet18().cuda()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.prepare_qat(model, inplace=True)

x = torch.rand(2, 3, 224, 224)
y0 = model(x.cuda())
y1 = model.to(device=torch.device("cpu"))(x)
print(y0)
print(y1)

import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)

model = torchvision.models.quantization.resnet.resnet18().cuda()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
torch.quantization.prepare_qat(model, inplace=True)

x = torch.rand(2, 3, 224, 224)
y0 = model(x.cuda())
y1 = model.to(device=torch.device("cpu"))(x)
print(y0)
print(y1)

import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)

model = torchvision.models.quantization.mobilenet_v2().cuda()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.prepare_qat(model, inplace=True)

x = torch.rand(2, 3, 224, 224)
y0 = model(x.cuda())
y1 = model.to(device=torch.device("cpu"))(x)

print(torch.nn.functional.cosine_similarity(y0, y1.cuda()))

import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)

model = torchvision.models.quantization.mobilenet_v2().cuda()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.prepare_qat(model, inplace=True)

x = torch.rand(2, 3, 224, 224)
y0 = model(x.cuda())
y1 = model(x.cuda())
print(torch.nn.functional.cosine_similarity(y0, y1))

import torch
import torch.nn as nn
import torchvision

torch.manual_seed(0)

model = torchvision.models.quantization.resnet.resnet18().cuda()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.prepare_qat(model, inplace=True)

x = torch.rand(2, 3, 224, 224)
model(x.cuda())

model.apply(torch.quantization.disable_observer)

y0 = model(x.cuda())
y1 = model.to(device=torch.device("cpu"))(x)

print(torch.nn.functional.cosine_similarity(y0, y1.to(y0)))

import torch
import torchvision
import copy

torch.manual_seed(0)

model = torchvision.models.quantization.resnet.resnet18().cuda()
model.train()
model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.prepare_qat(model, inplace=True)

x = torch.rand(2, 3, 224, 224)
model(x.cuda())
model1 = copy.deepcopy(model)

y0 = model(x.cuda())
y1 = model1.to(device=torch.device("cpu"))(x)

print(torch.nn.functional.cosine_similarity(y0, y1.to(y0)))


model2 = torchvision.models.quantization.resnet.resnet18()
model2.train()
model2.fuse_model()
model2.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
torch.quantization.prepare_qat(model2, inplace=True)

x = torch.rand(2, 3, 224, 224)
model2(x)
model3 = copy.deepcopy(model2)

y2 = model2(x)
y3 = model3(x)

print(torch.nn.functional.cosine_similarity(y2, y3))