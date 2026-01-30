import torch
import numpy as np

a = torch.zeros(3)
b = np.zeros(3)

a += b # works

a.copy_(b) # doesn't work
#   File "bug.py", line 8, in <module>
#    a.copy_(b) # doesn't work
#TypeError: copy_(): argument 'other' (position 1) must be Tensor, not numpy.ndarray