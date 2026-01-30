import numpy as np

import torch, numpy as np

a = torch.eye(2, 2).cuda()
print(np.linalg.inv(a))
# [[1. 0.]
# [0. 1.]]

print(np.linalg.inv(a.numpy()))
# TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.

print(torch.__version__)
# '0.5.0a0+bcd20f9'

def to_numpy(array):
    if isinstance(array, torch.Tensor):
        array = array.cpu()

    # torch only supports np.asarray for cpu tensors
    return np.asarray(array)