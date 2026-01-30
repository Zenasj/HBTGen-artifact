Python
import torch
import numpy as np
import time

with torch.no_grad():
    x_torch = [torch.rand(1,3,1000, 1200) for _ in range(10)]
    start = time.time()
    for i in range(100):
        torch.stack(x_torch)
    duration = time.time() - start
    print("Pytorch stack: ", duration)

    x_torch = [torch.rand(1,3,1000, 1200) for _ in range(10)]
    start = time.time()
    for i in range(100):
        torch.from_numpy(np.stack([x.numpy() for x in x_torch]))
    duration = time.time() - start
    print("NumPy stack: ", duration)

    x1 = torch.stack(x_torch)
    x2 = torch.from_numpy(np.stack([x.numpy() for x in x_torch]))
    is_equal = torch.equal(x1, x2)
    print("Is equal: ", is_equal)

torch.get_num_interop_threads()
torch.get_num_threads()