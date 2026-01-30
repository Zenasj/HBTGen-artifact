import torch.nn as nn

import torch, torch.nn as nn
import memory_profiler

@memory_profiler.profile

def main():
    model = nn.GRU(1000, 1000)
    x = torch.randn(10, 10, 1000)
    h = torch.randn(1, 10, 1000)
    model = model.cuda()
    x = x.cuda()
    h = h.cuda()
    model(x, h)

if __name__ == '__main__':
    main()