import torch
for i in range(1000):
    d = torch.float32.to_complex()

import torch
import gc
import sys

for i in range(50):
    d = torch.float32.to_complex()
    print(i, sys.getrefcount(d))
print("That's all folks!")
print(gc.get_referrers(d))