import torch.nn as nn

import torch 
input_tensor = torch.rand(1, 1, 480, 640).cuda()
coords = torch.FloatTensor([[-10059144, 67680944], [67680944, 67680944]]).unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1).cuda()
result = torch.nn.functional.grid_sample(input_tensor, coords)
print(result)