import torch
import numpy as np

3
b = np.zeros(8127, dtype=np.float32) # fake array

# Works
torch.tensor(b[:-1]) 
torch.tensor(b[1:])
torch.from_numpy(b) # no copying, new tensor and b share the same storage
torch.as_tensor(b) # no copying, new tensor and b share the same storage
torch.as_tensor(b, device='cuda')
torch.tensor(b.copy()) # creating a copy solves it but why?

# Any of this lines causes a segmentation fault and kills the kernel
torch.tensor(b)
torch.tensor(b, device='cpu', dtype=torch.float32, requires_grad=False)
torch.tensor(b[:len(b)])
torch.from_numpy(b).clone()
torch.zeros(8127).copy_(torch.from_numpy(b)) # a similar line is executed inside load_state_dict
torch.tensor(b.base)
b.setflags(write=True); torch.tensor(b) # with numpy 1.15.0