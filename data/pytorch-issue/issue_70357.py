import torch
a = torch.sparse_coo_tensor(
  indices=torch.tensor([[1,2,3]]),
  values=torch.tensor([3., 4., 5.], requires_grad=True),
  size=(1000,)
)

a.requires_grad
# True
a.coalesce().values().requires_grad
# True

# When we call without coalescing, gradients are lost
a._values().requires_grad
# False

import torch

def filter(x):
  # Let's filter some stuff!
  mask = x.values() > 0.5
  y_idx = x.indices()[:,mask]
  y_val = x.values()[mask]
  # Normally, I would have to coalesce y here, even though I'm guaranteed to not have any duplicate indices
  y = torch.sparse_coo_tensor(y_idx, y_val)
  return y

# Warm the cache
x = torch.sparse_coo_tensor(torch.arange(100000).unsqueeze(0), torch.rand(100000), size=(int(1e8),)).coalesce()
filter(x)

# Let's do some timing
import time
s0 = time.time() 
filter(x) 
s1 = time.time()
print(s1 - s0)

s0 = time.time() 
filter(x).coalesce()
s1 = time.time()
print(s1 - s0)

x = torch.sparse_coo_tensor(torch.arange(10).unsqueeze(0), torch.rand(10), size=(int(1e8),)).coalesce()
# Whoa, this won't fit in memory!
x.to_dense()
# Let's decrease the size
y = torch.sparse_coo_tensor(x.indices(), x.values(), size=(100,))
# Now we have to coalesce y even though the indices and values have not changed :(