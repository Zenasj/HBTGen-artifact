import torch
import torch.nn as nn


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

  def forward(self, theta, size):
    return torch.nn.functional.affine_grid(theta, size, align_corners=None)


model = Model()
theta = torch.ones((1, 2, 3))
size = torch.Size((1,3,24,24))
torch.onnx.export(model, (theta, size), 'test.onnx', verbose=True)

# https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html
# https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/AffineGridGenerator.cpp
def affine_grid(theta, size, align_corners=False):
    N, C, H, W = size
    grid = create_grid(N, C, H, W)
    grid = grid.view(N, H * W, 3).bmm(theta.transpose(1, 2))
    grid = grid.view(N, H, W, 2)
    return grid

def create_grid(N, C, H, W):
    grid = torch.empty((N, H, W, 3), dtype=torch.float32)
    grid.select(-1, 0).copy_(linspace_from_neg_one(W))
    grid.select(-1, 1).copy_(linspace_from_neg_one(H).unsqueeze_(-1))
    grid.select(-1, 2).fill_(1)
    return grid
    
def linspace_from_neg_one(num_steps, dtype=torch.float32):
    r = torch.linspace(-1, 1, num_steps, dtype=torch.float32)
    r = r * (num_steps - 1) / num_steps
    return r

def patch_affine_grid_generator():
    torch.nn.functional.affine_grid = affine_grid

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, theta, size):
        return torch.nn.functional.affine_grid(theta, size, align_corners=None)


model = Model()
theta = torch.ones((1, 2, 3))
size = torch.Size((1, 3, 24, 24))
torch.onnx.export(model, (theta, size), "test.onnx", opset_version=17, verbose=True)

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, theta, size):
        return torch.nn.functional.affine_grid(theta, size, align_corners=None)

model = Model()
theta = torch.ones((1, 2, 3))
size = torch.Size((1, 3, 24, 24))

torch.onnx.dynamo_export(model, theta, size).save("./b.onnx")

import torch
import torch.nn as nn


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

  def forward(self, theta, size):
    return torch.nn.functional.affine_grid(theta, size, align_corners=None)


model = Model()
theta = torch.ones((1, 2, 3))
size = torch.Size((1,3,24,24))
torch.onnx.export(model, (theta, size,), 'test.onnx', dynamo=True)