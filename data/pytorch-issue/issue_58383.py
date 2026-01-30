torch.testing.assert_close(1, 2)

print(0.00000001)
print(1e-1)

import torch

a = torch.rand(2, 3, 4, 5)
b = a.clone()
b[0, 1, 2, 3] *= 2

torch.testing.assert_close(a, b)

x = 0.123456789
print(f"{x:e}")
print(f"{x:f}")
print(f"{x:.9f}")