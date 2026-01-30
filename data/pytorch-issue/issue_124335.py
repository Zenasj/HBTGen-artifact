import torch

t = torch.ones((2**30-1,), dtype=torch.float32)
t2 = t.to("mps")
print("CPU <4GiB:", (t == torch.tensor(1)).all())
print("MPS <4GiB:", (t2 == torch.tensor(1, device="mps")).all())
print()

t = torch.ones((2**30,), dtype=torch.float32)
t2 = t.to("mps")
print("CPU 4GiB:", (t == torch.tensor(1)).all())
print("MPS 4GiB:", (t2 == torch.tensor(1, device="mps")).all())

import torch

t = torch.ones((2**30-1,), dtype=torch.float32)
t2 = t.to("mps")
print("CPU <4GiB:", (t == torch.tensor(1)).all())
print("MPS <4GiB:", (t2 == torch.tensor(1, device="mps")).all())
print()

t = torch.ones((2**30,), dtype=torch.float32, device="mps")
print (t)
print("CPU 4GiB:", (t == torch.tensor(1)).all())
print("MPS 4GiB:", (t2 == torch.tensor(1, device="mps")).all())