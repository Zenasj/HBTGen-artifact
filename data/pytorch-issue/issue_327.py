import torch
a = torch.IntTensor([0,1])
b = torch.IntTensor([0,1])
print(a.div(b)) # Floating point exception (core dumped)

import torch
a = torch.FloatTensor([1,1])
b = torch.FloatTensor([0,0])
a.div(b)