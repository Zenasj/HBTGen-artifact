import torch.nn as nn

import torch
import torch.nn.functional as F

dev_cpu = torch.device('cpu')
dev_mps = torch.device('mps')

device=dev_cpu

laf = torch.tensor([[2, 0, 4.], [0, 2., 5.]], device=device)
grid = F.affine_grid(laf.view(1,2,3), [1, 3, 3, 3], align_corners=False)

print ("cpu grid:", grid)

laf=laf.to(dev_mps)
grid = F.affine_grid(laf.view(1,2,3), [1, 3, 3, 3], align_corners=False)
print ("mps grid:", grid)

import torch
import torch.nn.functional as F

dev_cpu = torch.device('cpu')
dev_mps = torch.device('mps')

device=dev_cpu

laf = torch.tensor([[2, 0, 4.], [0, 2., 5.]], device=device)
grid_cpu = F.affine_grid(laf.view(1,2,3), [1, 3, 3, 3], align_corners=False)

print ("cpu grid:", grid_cpu)

laf=laf.to(dev_mps)
grid_mps = F.affine_grid(laf.view(1,2,3), [1, 3, 3, 3], align_corners=False)
print ("mps grid:", grid_mps)

print(f"{((grid_cpu-grid_mps.cpu()).abs() > 1e-7).sum() = }")

# Output:
# cpu grid: tensor([[[[2.6667, 3.6667],
#           [4.0000, 3.6667],
#           [5.3333, 3.6667]],
# 
#          [[2.6667, 5.0000],
#           [4.0000, 5.0000],
#           [5.3333, 5.0000]],
# 
#          [[2.6667, 6.3333],
#           [4.0000, 6.3333],
#           [5.3333, 6.3333]]]])
# mps grid: tensor([[[[2.6667, 3.6667],
#           [4.0000, 3.6667],
#           [5.3333, 3.6667]],
# 
#          [[2.6667, 5.0000],
#           [4.0000, 5.0000],
#           [5.3333, 5.0000]],
# 
#          [[2.6667, 6.3333],
#           [4.0000, 6.3333],
#           [5.3333, 6.3333]]]], device='mps:0')
# ((grid_cpu-grid_mps.cpu()).abs() > 1e-7).sum() = tensor(0)