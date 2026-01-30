from torch.distributed.distributed_c10d import _object_to_tensor
import time

start = time.time()
_object_to_tensor("x" * 50_000_000)
print("Time:", time.time() - start)

import torch
import io
import time
import pickle

f = io.BytesIO()
pickle.dump("x" * 50_000_000, f)

byte_storage = torch.ByteStorage.from_buffer(f.getvalue())

start = time.time()
byte_tensor = torch.tensor(byte_storage, dtype=torch.uint8)
print("torch.tensor time:", time.time() - start)

start = time.time()
byte_tensor = torch.ByteTensor(byte_storage)
print("torch.ByteTensor time:", time.time() - start)