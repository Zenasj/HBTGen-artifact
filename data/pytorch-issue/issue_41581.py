import torch
import numpy as np
a = torch.rand(5)
b = np.arange(5)
a[torch.arange(2)] = torch.tensor(b[:2])
print(a)

import torch
import numpy as np
a = torch.rand(5)
b = np.arange(5)
a[torch.arange(2)] = torch.tensor(b[:2]).to(a.dtype)
print(a)