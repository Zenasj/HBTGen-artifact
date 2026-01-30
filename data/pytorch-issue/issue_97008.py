import torch.nn as nn

import torch
import torch._dynamo

class TestSig(torch.nn.Module):
   def __init__(self):
      super().__init__()
   def forward(self, x):
      return torch.sigmoid(x)

torch._dynamo.config.verbose=True
opt_cpu = torch.compile(TestSig())
print("cpu:", opt_cpu(torch.randn(1)))
cuda_eager = TestSig().cuda()
print("cuda eager:", cuda_eager(torch.randn(1).cuda()))
opt_cuda = torch.compile(TestSig()).cuda() #torch.compile(TestSig().cuda()) also fails
print("cuda opt:", opt_cuda(torch.randn(1).cuda()))