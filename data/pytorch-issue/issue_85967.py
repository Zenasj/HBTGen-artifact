import torch
import numpy as np

a = np.array(np.arange(9).reshape((3,3)))
a = a[:,:2]
a_mps = torch.tensor(a, device=torch.device('mps'))
a_cpu = torch.tensor(a, device=torch.device('cpu'))

print(a)
# [[0 1]
#  [3 4]
#  [6 7]]

print(a_cpu)
# tensor([[0, 1],
#         [3, 4],
#         [6, 7]])  -- CORRECT

print(a_mps)
# tensor([[0, 1],
#         [2, 3],
#         [4, 5]], device='mps:0') -- INCORRECT