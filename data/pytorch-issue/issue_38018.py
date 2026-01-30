import torch
import numpy as np

x = torch.full((1, 3), np.nan, dtype=torch.float64).cuda()
x[:, :1] = 1.1
r = torch.median(x, dim=1).values
print(r)
r = torch.median(x.cpu(), dim=1).values
print(r)

import torch
import numpy as np

print(torch.__version__)
print(np.__version__)

device = 'cuda'

x = torch.tensor([
    [np.nan, 1.1, np.nan],
    [np.nan, 1.1, np.nan]
], dtype=torch.float, device=device)

# print(x)

a = torch.median(x, dim=1)
print(a)

import torch
import numpy as np

print(torch.__version__)
print(np.__version__)

device = 'cuda'

x = torch.tensor([
    [np.nan, 1.1, np.nan],
    [np.nan, 1.1, np.nan]
], dtype=torch.float, device=device)

print(x)    # Note: a print(x) here!!

a = torch.median(x, dim=1)
print(a)