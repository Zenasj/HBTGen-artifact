import torch.nn as nn
import random

import numpy as np
import os
import time
import PIL
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.resnet   import  ResNet, BasicBlock

batch_size = 128
max_number_of_epoch = 100
LR = 0.1
image_size = 224
number_of_classes = 10
clipping_value = 512

np.random.seed(2)
torch.manual_seed(2)

n_cpu = n_cpu = int(os.cpu_count()*0.8)

class Average_Meter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0

  def update(self, val, n):
    if n > 0:
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

class Sum_Meter(object):
  """Computes and stores the sum and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0

  def update(self, val, n):
    if n > 0:
      self.val = val
      self.sum += val
      self.count += n
      self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


transform = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=n_cpu)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

class MyResNet18(ResNet):
  def __init__(self, num_classes):
    super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)
      
  def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    FV = torch.flatten(x, 1)
    Logit = self.fc(FV)
    return FV, Logit


model = MyResNet18(num_classes=number_of_classes) 

best_val = 0.0

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model.to(device)

loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()

torch.backends.cudnn.benchmark=True

losses_train_total = Sum_Meter()
top1_train = Average_Meter()
top5_train = Average_Meter()
losses_val_total = Sum_Meter()
top1_val = Average_Meter()
top5_val = Average_Meter()

for epoch in range(max_number_of_epoch):
  print("\nepoch = ", epoch +1)
  losses_train_total.reset()
  top1_train.reset()
  top5_train.reset()
  losses_val_total.reset()
  top1_val.reset()
  top5_val.reset()
  t1 = time.time()
  model.train()
  for i, (x,y) in enumerate(train_loader, 0):
    optimizer.zero_grad()
    x = x.cuda()
    y = y.detach().clone().long().cuda()
    FV, Logit  = model(x)
    prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    loss = loss_CE(Logit,y)
    losses_train_total.update(loss.item(), y.size(0))
    top1_train.update(prec1.item(), y.size(0))
    top5_train.update(prec5.item(), y.size(0))
    assert not torch.isnan(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
    optimizer.step()
  model.eval()
  with torch.no_grad():
    for i, (x,y) in enumerate(val_loader, 0):
      x = x.cuda()
      y = y.detach().clone().long().cuda()
      FV, Logit  = model(x)
      prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
      loss = loss_CE(Logit,y)
      losses_val_total.update(loss.item(), y.size(0))
      top1_val.update(prec1.item(), y.size(0))
      top5_val.update(prec5.item(), y.size(0))
  t2 = time.time()
  print('train average_loss_total', losses_train_total.avg)
  print('train top1 accuracy', top1_train.avg)
  print('train top5 accuracy ', top5_train.avg)
  print('validation average_loss_total', losses_val_total.avg)
  print('validation top1 accuracy', top1_val.avg)
  print('validation top5 accuracy ', top5_val.avg)
  print("epoch time = ", t2-t1)
  if top1_val.avg > best_val:
    best_val = top1_val.avg
    print("model saved with vallidation top-1 accuracy  =  " , best_val)
    torch.save(model.state_dict(), f"resnet_18_cifar_10_vall_acc_{best_val}.pth.tar")

print('Finished Training')

import numpy as np
import os
import time
import PIL
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.resnet   import  ResNet, BasicBlock

batch_size = 32
max_number_of_epoch = 100
LR = 0.1
number_of_classes = 10
clipping_value = 512

np.random.seed(2)
torch.manual_seed(2)


class MyResNet18(ResNet):
  def __init__(self, num_classes):
    super(MyResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes = num_classes)
      
  def _forward_impl(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    FV = torch.flatten(x, 1)
    Logit = self.fc(FV)
    return FV, Logit

model = MyResNet18(num_classes=number_of_classes) 

best_val = 0.0

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model.to(device)

loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()

torch.backends.cudnn.benchmark=True

model.train()
for epoch in range(max_number_of_epoch):
  print("\nepoch = ", epoch +1)
  t1 = time.time()
  for i in range(100):
    print("\niteration = ", i)
    optimizer.zero_grad()
    x = torch.randn(batch_size, 3, 224, 224).cuda()
    y = torch.randint(low = 0, high = number_of_classes, size = (batch_size,)).long().cuda()
    print("before model(x)")
    FV, Logit  = model(x)
    print("after model(x)")
    loss = loss_CE(Logit,y)
    assert not torch.isnan(loss)
    print("after loss_CE")
    loss.backward()
    print("after loss.backward()")
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
    print("after clip_grad_norm_")
    optimizer.step()
    print("after optimizer.step()")
  t2 = time.time()
  print("epoch time = ", t2-t1)

print('Finished Training')

tf.distribute.MirroredStrategy