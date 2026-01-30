import torch

import torch.nn as nn
import copy
x = torch.randn(1, 1, 80, 80)

m_fp32 = nn.Conv2d(1, 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
m_bf16 = copy.deepcopy(m_fp32).bfloat16()

x_bf16 = x.bfloat16()

y_fp32 = m_fp32(x).sum()
y_fp32.backward()

y_bf16 = m_bf16(x_bf16).sum()
y_bf16.backward()

print(m_fp32.bias.grad)
print(m_bf16.bias.grad)

tensor([1600., 1600.])
tensor([256., 256.], dtype=torch.bfloat16)

tensor([1600., 1600.])
tensor([1600., 1600.], dtype=torch.bfloat16)