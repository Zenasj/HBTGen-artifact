import torch.nn as nn

import torch
results = dict()

input1_tensor = torch.randint(0, 8, [0, 20], dtype=torch.uint8)
input2_tensor = torch.rand([0, 30], dtype=torch.float32)
weight_tensor = torch.rand([40, 20, 30], dtype=torch.float32)
bias_tensor = torch.rand([40], dtype=torch.float32)

input1 = input1_tensor.clone()
input2 = input2_tensor.clone()
weight = weight_tensor.clone()
bias = bias_tensor.clone()

res1 = torch.nn.functional.bilinear(input1, input2, weight, bias=bias, )
# Normal Pass

input1 = input1_tensor.clone()
input2 = input2_tensor.clone().requires_grad_()
weight = weight_tensor.clone().requires_grad_()
bias = bias_tensor.clone().requires_grad_()
res2 = torch.nn.functional.bilinear(input1, input2, weight, bias=bias, )
# RuntimeError: isDifferentiableType(variable.scalar_type())INTERNAL ASSERT FAILED at "/Users/distiller/project/pytorch/torch/csrc/autograd/functions/utils.h":65, please report a bug to PyTorch.