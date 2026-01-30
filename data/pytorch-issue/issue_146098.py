import torch
t1 = torch.empty([2,4], pin_memory=True)
assert t1.is_pinned() # Works correctly

import torch
t2 = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 8]])
t2 = t2.pin_memory() # lazyInitDevice is not called
assert t2.is_pinned() # t2.is_pinned() == False