import torch
import numpy as np

1, 1, 1, 1
0, 1, 1, 1
1, 1, 1, 1

0
1
0

0
2
0

x = [[1, 4, 4, 3], [5, 5, 2, 5]]
n = 4  # the size of the dimension where we want to take the argmax

# Need to map the flipped indexs to the original ones.
n - 1 - torch.tensor(x).flip(dims=(1,)).argmax(dim=1)

x = [[1, 4, 4, 3], [5, 5, 2, 5]]
np.argmax(np.array(x), axis=1)