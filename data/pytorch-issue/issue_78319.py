import torch
torch.tensor(True, device='mps')
print(torch.full((5,), False, device='mps'))

import torch as pt

print(hex(pt.tensor(1, device='mps').data_ptr()))
print(hex(pt.tensor(True, device='mps').data_ptr()))
# x will be tensor([ True, False, False, False, False, False, False, False, False, False], device='mps:0')
# ==> First element corrupted in both cases!

print(hex(pt.tensor(0.1, device='mps').data_ptr()))
# x will be tensor([ True, False, False,  True, False, False, False, False, False, False], device='mps:0')
# ==> Corruption happens for any datatype, memory seems "mixed"

s = pt.tensor(1, device='mps')
print(hex(s.data_ptr()), s)
# x will not be corrupted
# ==> Memory is retained by binding to `s`, and no corruption happens!
#     This could mean that Tensors are not freed correctly.

s = pt.tensor(1, device='mps')
print(hex(s.data_ptr()), s)
del(s)
# x will be tensor([ True, False, False, False, False, False, False, False, False, False], device='mps:0')
# ==> Corruption again! Freeing scalar tensors causes corruption.

print(hex(pt.tensor(True).data_ptr()))
# x will not be corrupted
# ==> Corruption only happens via MPS scalar tensors

x = pt.full((10,), False, device="mps")  # this one gets corrupted
# x = pt.full((10,), 0, device="mps")    # no corruption (dtype != bool)
# x = pt.full((10,), 0.1, device="mps")  # no corruption (dtype != bool)
print(hex(x.data_ptr()), x)  # identical memory location as the scalar tensor!
if any(x):
    print('Tensor corrupted')