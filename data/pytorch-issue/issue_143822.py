import random

import time
import torch
import numpy as np
np.random.seed(2024)

op_list = [torch.floor_divide, torch.fmod, torch.remainder, torch.divide]
cop_list = [torch.compile(f) for f in op_list]
dtype_list = [torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64]

# cold start
for cop in cop_list:
    for dtype in dtype_list:
        value = torch.tensor(np.random.randn(1,2,3), dtype=dtype)
        value[value == 0] = 1
        divisor = torch.tensor(np.random.randn(1,2,3), dtype=dtype)
        divisor[divisor == 0] = 1
        try:
            res = cop(value, divisor)
        except RuntimeError as e:
            pass

for op, cop in zip(op_list, cop_list):
    print(f"Benchmarking {op.__name__}")
    inference_time_list = []
    for dtype in dtype_list:
        for i in range(100):
            value = torch.tensor(np.random.randn(1,2,3), dtype=dtype)
            value[value == 0] = 1
            divisor = torch.tensor(np.random.randn(1,2,3), dtype=dtype)
            divisor[divisor == 0] = 1
            start = time.time()
            try:
                res = cop(value, divisor)
            except RuntimeError as e:
                pass
            inference_time_list.append(time.time() - start)
    print(f"Average inference time: {np.mean(inference_time_list)}")
    print(f"Max inference time: {np.max(inference_time_list)}")
    print(f"Min inference time: {np.min(inference_time_list)}")