import torch
import numpy as np

# Create a torch.int tensor from numpy
a_np = np.asarray([0, 1, 1, 0, 0])
a = torch.from_numpy(a_np)

# Create a torch.float tensor directly
b = torch.FloatTensor([0, 1, 1, 0, 0])

# Perform torch.ge on a and b
print(torch.ge(a, 0.5))   # Output: tensor([ 1,  1,  1,  1,  1], dtype=torch.uint8)
print(torch.ge(b, 0.5))   # Output: tensor([ 0,  1,  1,  0,  0], dtype=torch.uint8)