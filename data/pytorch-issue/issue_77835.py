import torch
import numpy as np

test = torch.from_numpy(np.asarray([True, True, True, False])).to('mps')
print(test)

print(test.type(torch.float))

test = torch.from_numpy(np.asarray([True, True, True, False])).to('cpu')
print(test)

print(test.type(torch.float))