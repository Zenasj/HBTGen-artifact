import torch
import numpy as np
A = np.nan * torch.ones((3,3))
torch.linalg.eig(A)

with torch.linalg.nancheck(Flase):
  # This will run faster but might crash
   torch.linalg.eig(torch.linalg.eig(float('nan') * torch.ones(3, 3)))