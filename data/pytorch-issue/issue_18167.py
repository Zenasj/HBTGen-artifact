import random

python
import torch
import numpy as np
import math

def get_mean(dtype, dev):
    try:
        data = np.random.randint(10, 125, size=(10, 20, 30))
        x = torch.tensor(data, dtype=dtype).to(dev)
        t_mean = x.mean()
        if not math.isclose(t_mean, np.mean(data)):
            # Print type and values if they are not
            # close to numpy mean.
            # print(x.dtype, t_mean, t_mean.dtype , np.mean(data))
            print("For {} and device {} the numpy output wasn't close".format(dtype, dev))
    except Exception as e:
        print(e)
    
dtypes = [torch.float16, torch.float32, torch.float64, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
for d in dtypes:
    get_mean(d,dev='cuda')
    get_mean(d,dev='cpu')