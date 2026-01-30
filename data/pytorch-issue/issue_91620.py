import torch as th
# this works
a = th.eye(9, device="mps").bool()
print(a.shape)
# this fails
th.eye(9, dtype=bool, device="mps")