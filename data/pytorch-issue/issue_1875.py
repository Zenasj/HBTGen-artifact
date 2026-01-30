import torch
import numpy as np
import random

def get_mean(dtype):
    try:
        data = np.random.randint(10, size=10)
        x = torch.tensor(data, dtype=dtype)
        print(x.dtype, x.mean())
    except Exception as e:
        print(e)
    
dtypes = [torch.float16, torch.float32, torch.float64, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
for d in dtypes:
    get_mean(d)