import torch.nn as nn

import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data.distributed
import torchvision.models as models

train_dataset = \
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((224, 224)),
                       transforms.Lambda(lambda image: image.convert('RGB')),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                   ]))

torch.cuda.set_device(0)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=2)

# hyper parameter
epoch_ = 100
lr_ = 0.001
momentum_ = 0.9
milestones = [30, 80, 120]
log_interval = 10

model = models.alexnet(num_classes=10)
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr_, momentum=momentum_)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    ) as p: 
    for epoch in range(0, epoch_):
        model.train()
        for step, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, step * len(data), len(train_loader),
                    100.0 * step / len(train_loader), loss.item()))
            p.step()
        scheduler.step()

import numpy as np
import torch
from torchvision.models import resnet18

if __name__ == '__main__':
    model = resnet18(pretrained=False)
    device = torch.device('cuda')
    model.to(device)
    
    BATCH_SIZE = 64
    BATCH_NUM = 10
    train_loader = [[torch.rand(BATCH_SIZE,3,224,224),torch.randint(0,10,[BATCH_SIZE])]
                      for i in range(BATCH_NUM)]
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=3,repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("torchprofilertest"),
        with_stack=True,
        # record_shapes=True,
        # profile_memory=True,
        # with_flops=True,
        # with_modules=True,
    ) as p:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            # torch.cuda.synchronize()
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(batch_idx,0)
            p.step()
            print(batch_idx,1)