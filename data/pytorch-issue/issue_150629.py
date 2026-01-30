import torch.nn as nn

import torch
with torch.no_grad():
    x = torch.randn(1024, requires_grad=False, device="mps")
    model = torch.nn.RMSNorm(1024, device="mps")
    y1 = model(x)
    model = torch.compile(model)
    y2 = model(x)
    print(y1)
    print(y2)