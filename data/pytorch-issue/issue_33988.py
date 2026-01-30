import torch
import torch.nn as nn
problem_tensor = torch.load("tensorProblem.pt")
max_pooled = torch.max_pool2d(problem_tensor, 2, 2, 0, 1, False)

import torch
import torch.nn as nn
problem_tensor = torch.load("tensorProblem.pt")
tmp = torch.zeros(problem_tensor.size())
tmp[:,:,:,:] = problem_tensor[:,:,:,:]
max_pooled = torch.max_pool2d(tmp, 2, 2, 0, 1, False)