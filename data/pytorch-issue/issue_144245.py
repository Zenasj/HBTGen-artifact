import torch.nn as nn

import torch
import torch.nn.functional as F


test = torch.Tensor([[1],[2],[4]]).to("mps")
result = F.interpolate(test.unsqueeze(1), 3, mode="linear", align_corners=True).squeeze(1)

print(result)
# tensor([[nan, nan, nan],
#         [nan, nan, nan],
#         [nan, nan, nan]], device='mps:0')
test = torch.Tensor([[1],[2],[4]]).to("cpu")
result = F.interpolate(test.unsqueeze(1), 3, mode="linear", align_corners=True).squeeze(1)

print(result)
# tensor([[1., 1., 1.],
#         [2., 2., 2.],
#         [4., 4., 4.]])