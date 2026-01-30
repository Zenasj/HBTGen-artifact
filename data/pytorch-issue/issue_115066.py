import random

import numpy as np
import torch
x = torch.tensor([[1.0]], dtype=torch.uint8)
y = torch.tensor(np.random.rand(0,0), dtype=torch.bfloat16)
res = x*y

import numpy as np
import torch
x = torch.tensor([[1.0]], dtype=torch.float16)
y = torch.tensor(np.random.rand(0,0), dtype=torch.float16)
res = x*y

import numpy as np
import torch
x = torch.tensor([[1.0]], dtype=torch.float16)
y = torch.tensor(np.random.rand(0,0), dtype=torch.float16)
res = x*y

import numpy as np
import torch
x = torch.tensor([[1.0]], dtype=torch.float16)
y = torch.tensor(np.random.rand(0,0), dtype=torch.int64)
res = x*y