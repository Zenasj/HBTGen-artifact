import torch.nn as nn

import torch
torch.nn.functional.conv1d(input=torch.ones((1,1,1)),weight=torch.ones((1,1,1)),groups=0)

import torch
torch.nn.functional.conv2d(input=torch.ones((1,1,1,1)),weight=torch.ones((1,1,1,1)),groups=0)

import torch
torch.nn.functional.conv3d(input=torch.ones((1,1,1,1,1)),weight=torch.ones((1,1,1,1,1)),groups=0)

torch.nn.functional.conv_transpose1d(input=torch.ones((1,1,1)),weight=torch.ones((1,1,1)),groups=0)
torch.nn.functional.conv_transpose2d(input=torch.ones((1,1,1,1)),weight=torch.ones((1,1,1,1)),groups=0)
torch.nn.functional.conv_transpose3d(input=torch.ones((1,1,1,1,1)),weight=torch.ones((1,1,1,1,1)),groups=0)

py
TORCH_CHECK(groups > 0, error_message)