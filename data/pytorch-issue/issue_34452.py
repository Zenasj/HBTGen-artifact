import torch
import numpy as np

a=np.array([0.])
t=torch.tensor([0.])
i=np.array([[0,0],[0,0]])
a[i] # Fine
t[torch.tensor(i)] # Fine
t[i] # IndexError: too many indices for tensor of dimension 1