import torch.nn as nn

import torch
import torch._inductor.config

torch._inductor.config.worker_start_method = 'spawn'
torch._inductor.config.compile_threads = 2
from torch._inductor.codecache import AsyncCompile


if __name__ == '__main__':
    model = torch.nn.Linear(10,10).cuda()
    model = torch.compile(model)
    inp = torch.rand(10, 10).cuda()
    model(inp).sum().backward()
    print("Done")