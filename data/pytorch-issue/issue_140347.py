import torch

a = torch.tensor(float('nan'))
b = a.cpu().type(torch.int32)
c = a.cuda().type(torch.int32)
print(b)  # tensor(-2147483648, dtype=torch.int32)
print(c)  # tensor(0, device='cuda:0', dtype=torch.int32)

import torch

a = torch.tensor(float('nan'))

b = a.cpu().type(torch.int64)
c = a.cuda().type(torch.int64)

print(b)  # tensor(-9223372036854775808)
print(c)  # tensor(-9223372036854775808, device='cuda:0')