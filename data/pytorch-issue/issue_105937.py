import torch
import torch.nn as nn

py
x = torch.randn(3,3,16,16)
norm = torch.nn.BatchNorm2d(3, track_running_stats=True)
my_var = torch_norm.running_var.clone()
for i in range(3):
   my_var = (1 - norm.momentum) * my_var + norm.momentum * (x ** i).var((0,2,3), unbiased=True)
   out = norm(x ** i)
   print(f"{my_var=}, {norm.running_var=}")