torch.Size([2, 2, 1])
tensor([[[[[1],
           [1]]],

         [[[1],
           [1]]]],

        [[[[1],
           [1]]],

         [[[1],
           [1]]]]])

import torch
import torch.nn as nn
data=torch.tensor([[[1],[1]],[[1],[1]]]);
index=torch.tensor([[[1],[1]],[[1],[1]]]);
print(data.shape)
print(data[index])

data=torch.tensor([[[1],[1]],[[1],[1]]]);
index=torch.tensor([[[1],[1]],[[1],[1]]]);
print(data[index])