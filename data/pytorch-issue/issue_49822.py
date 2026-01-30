import torch
import numpy as np

a = torch.tensor([-57.0625], dtype=torch.half)
b = torch.tensor([1.2339], dtype=torch.half)

print(torch.remainder(a,b))

an = a.numpy()
bn = b.numpy()

print(np.remainder(an, bn))


# output
tensor([0.9375], dtype=torch.float16)  # cpu
[0.953] # numpy