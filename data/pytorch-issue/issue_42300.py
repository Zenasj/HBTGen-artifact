import torch.nn as nn

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 4
        self.cls_subnet = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ])
        self.bbox_subnet = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ])

    def forward(self, x):
        cls_subnet = self.cls_subnet(x)
        bbox_subnet = self.bbox_subnet(x)
        return cls_subnet, bbox_subnet


if __name__ == "__main__":
    x = torch.randn(2, 4, 100, 152)
    x.requires_grad_()
    head = Head()
    cls, box = head(x)
    N, _, H, W = box.shape
    tensor = box.reshape(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
    tensor.mean().backward()

import torch
from torch import nn


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 4
        self.bbox_subnet = nn.Sequential(*[
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ])

    def forward(self, x):
        bbox_subnet = self.bbox_subnet(x)
        return bbox_subnet


if __name__ == "__main__":
    x = torch.randn(2, 4, 100, 152)
    x.requires_grad_()
    head = Head()
    box = head(x)
    N, _, H, W = box.shape
    tensor = box.reshape(N, -1, 4, H, W).permute(0, 3, 4, 1, 2)
    tensor.mean().backward()

model1 = build_model()  # on cpu
model1 = model1.cuda()  # moved to cuda device

ckpt_path = "./ckpt1.pt"
torch.save(model1.state_dict(), ckpt_path)  # model1 params on cuda 

model2 = build_model()  # on cpu
model2.load_state_dict(torch.load(ckpt_path))  # loading params original on cuda onto cpu model
model2 = model2.cuda()
train(model2)  # some function that does training

model1 = build_model()  # on cpu
model1 = model1.cuda()  # moved to cuda device

ckpt_path = "./ckpt1.pt"
torch.save(model1.state_dict(), ckpt_path)  # model1 params on cuda 

model2 = build_model()  # on cpu
model2 = model2.cuda()  # move to cuda before loading state dict
model2.load_state_dict(torch.load(ckpt_path))
...

def forward(self, u):
    u = u.permute(0,2,3,1).matmul(self.W0)
    u = u.permute(0,2,3,1).matmul(self.W1)
    u = u.permute(0,2,3,1).matmul(self.W2)
    return u