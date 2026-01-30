import torch

import time

from torch.distributed.distributed_c10d import _object_to_tensor

start = time.time()
_object_to_tensor("x" * 50_000_000)
print("Time:", time.time() - start)