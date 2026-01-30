import torch
import numpy as np
import random

temp = torch.tensor(np.ones((128,128,1)).astype(np.uint16).astype(np.int16), dtype=torch.float32, device="cpu")

a = np.random.randint(2**15, 2**16, dtype=np.uint16)    # A random uint16 number
t_int16 = torch.from_numpy(a.astype(np.int16)).cuda()   # Notice that only 16 bits are transferred
t_uint16 = t_int16.to(torch.int32) & (2**16 - 1)        # Extract required bits to match value store in "a"