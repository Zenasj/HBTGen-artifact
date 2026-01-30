import torch

def do(x):
  x_c = x.contiguous()
  x.logit_()
  x_c.logit_()
  print("x   %s" % x)
  print("x_c %s" %x_c)
  x_c = x.contiguous()
  print("xee  %s" % x.logit(1e-6))
  print("x_c  %s" % x_c.logit(1e-6))

x = torch.randn((1,2,2,8)).contiguous(memory_format=torch.channels_last)
do(x)