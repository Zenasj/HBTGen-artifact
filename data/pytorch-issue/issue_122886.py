import torch
import numpy as np
torch.set_default_device("cuda:0")

t = torch.tensor(1)
print("tensor:", t.device)
n = torch.normal(0.,1., (10,10))
print("normal:", n.device)
nt=torch.from_numpy(np.array([10.,20.]))
print("nt:", nt.device)