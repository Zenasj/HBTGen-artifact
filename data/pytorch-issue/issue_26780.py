import torch

# this works fine
A = torch.zeros(1, device="cuda")
A[[True]] = torch.ones(1)

# this works fine 
A = torch.zeros(1, device="cuda")
A[[True]] = torch.ones(1, device="cuda")

# this crashes
A = torch.zeros(1)
A[[True]] = torch.ones(1, device="cuda")

# this works fine
A = torch.zeros(1)
A[0] = torch.ones(1, device="cuda")

import torch
A = torch.zeros(1, device="cuda")
A[[True]] = torch.ones(1)
print(A)