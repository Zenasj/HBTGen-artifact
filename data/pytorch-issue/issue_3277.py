wrapped_x = np.array([1, -2, 3, -4], dtype=dtype)
print(wrapped_x)
print(wrapped_x[1])

import torch
import numpy as np

dtype = np.uint8 
x = torch.Tensor([1, -2, 3, -4])
asarray = np.asarray(x, dtype=dtype)
print ("asarray", asarray)
np_array = np.array(x, dtype=dtype)
print ("array x from torch", np_array)
array = np.array([1, -2, 3, -4], dtype=dtype)
print ("array x not from torch", array)