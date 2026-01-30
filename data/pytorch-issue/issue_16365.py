import torch.nn as nn

import torch
batch_size = 256*256
transform_parameters = torch.cuda.FloatTensor([[1,0,0],[0,1,0]])
transform_parameters = torch.stack([transform_parameters] * batch_size, 0).contiguous()
resampling_grids = torch.nn.functional.affine_grid(transform_parameters, torch.Size((batch_size, 1, 2, 2)))
print(resampling_grids.size())