import numpy as np
import random

import torch
for device, dtype in [('cuda:0', torch.float16), ('cuda:0', torch.float32), ('cpu', torch.float32)]:
        A = torch.randn(10_000, 10_000, dtype=dtype).to(device)
        B = torch.randn(10_000, dtype=dtype).to(device)
        C = torch.randn(10_000, dtype=dtype).to(device)

        result_1 = A @ B + A @ C
        result_2 = A @ (B + C)

        print(f'Device: {device}, dtype: {dtype}')
        print(f'Are all values close: {torch.allclose(result_1, result_2)}')
        print(f'The norm of the difference is: {(result_1 - result_2).norm()}')
        print('=' * 20)

A = np.random.randn(10_000, 10_000)
B = np.random.randn(10_000)
C = np.random.randn(10_000)

result_1 = A @ B + A @ C
result_2 = A @ (B + C)

print(f'Are all values close: {np.allclose(result_1, result_2)}')
print(f'The norm of the difference is: {np.linalg.norm(result_1 - result_2)}')