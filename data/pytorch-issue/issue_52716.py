import torch.nn as nn

import torch
x = torch.randn((2200000000, 1), device="cuda:0", dtype=torch.float16, requires_grad=True)
torch.cuda.profiler.start()
y = torch.nn.functional.log_softmax(x, dim=-1)
y.backward(y)
torch.cuda.profiler.stop()
torch.cuda.synchronize()