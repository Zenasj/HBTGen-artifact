import torch.nn as nn
import random

import torch
torch.random.manual_seed(420)
input = torch.randn(2,3,requires_grad=True)
res_cpu = torch.nn.functional.gumbel_softmax(input, hard=True)
print("res_cpu: ", res_cpu)
input2 = input.clone().detach().to('cuda')
res_gpu = torch.nn.functional.gumbel_softmax(input2, hard=True)
print("res_gpu: ", res_gpu)