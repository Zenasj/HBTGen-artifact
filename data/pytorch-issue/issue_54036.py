import torch
import torch.nn as nn
model = nn.Conv2d(64, 128, [1, 1], [2, 2], [0, 0])
x = torch.randn(1, 64, 8, 8)

model.cuda()
result_cudnn = model(x.cuda())

torch.backends.cudnn.enabled=False
result = model(x.cuda())

assert torch.allclose(result, result_cudnn)