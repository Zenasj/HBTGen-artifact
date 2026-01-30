3
import torch
import cupy as cp

cupy_arr = cp.asarray([1, 2, 3])

# Default case
t = torch.as_tensor(cupy_arr)
# New behavior, same device as cupy_arr now, was cpu before
print(t.device)  # cuda:0

# Explicitly set device
t = torch.as_tensor(cupy_arr, device='cpu')
print(t.device)  # cpu

# Implicit default device
torch.set_default_device('cpu')
t = torch.as_tensor(cupy_arr)
print(t.device)  # cpu

# Default device via context manager
torch.set_default_device('cuda')
with torch.device('cpu'):
    t = torch.as_tensor(cupy_arr)
    print(t.device)  # cpu

# Unset default device
torch.set_default_device(None)
t = torch.as_tensor(cupy_arr)
# New behavior, same device as cupy_arr now, was cpu before
print(t.device)  # cuda:0