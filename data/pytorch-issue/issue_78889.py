import torch
import numpy as np

x = np.array([True, False, True])
x.mean()
0.6666666666666666

x = torch.tensor([True, False, True])
x.sum().div(len(x))
tensor(0.6667)