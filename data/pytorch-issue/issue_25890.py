python
import torch
import torch.nn as nn
# Float works

X  = torch.rand(1, 1, 32,32)
X_q = torch.quantize_linear(X, 1.0, 0, torch.quint8)
conv1 = nn.Conv2d(1, 2, 5, 1)

y = conv1(X_q.dequantize())
z = torch.nn.functional.max_pool2d(y, 2, 2)
print(z.size())
z = z.view(-1,2*14*14)

torch.Size([1, 2, 14, 14])

# Quantized does not
conv1 = nn.quantized.Conv2d(1, 2, 5, 1)
y = conv1(X_q)
z = torch.nn.functional.max_pool2d(y, 2, 2)
print(z.size())
z = z.view(-1,2*14*14)