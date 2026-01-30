import torch.nn as nn

import torch
import torch.nn.functional as F

input_size = [2, 3, 0, 7]
reduction = "mean"
device = 'mps'

input_cpu = torch.rand(input_size, requires_grad=True, device='cpu')
input_mps = input_cpu.to(device)

num_channels = input_size[1]
target_size = (input_size[0], ) + tuple(input_size[2:])
target_cpu = torch.randint(num_channels, target_size, device='cpu')
target_mps = target_cpu.to(device)

output_cpu = F.nll_loss(input_cpu, target_cpu, reduction=reduction)
# tensor(nan, grad_fn=<NllLoss2DBackward0>)
output_mps = F.nll_loss(input_mps, target_mps, reduction=reduction)
# RuntimeError: [srcBuf length] > 0 INTERNAL ASSERT FAILED at "/Users/hvaara/dev/pytorch/pytorch/aten/src/ATen/native/mps/OperationUtils.mm":524, please report a bug to PyTorch. Placeholder tensor is empty!