import torch
import numpy as np

assert_close(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [4.0, 4.0]]) # compare 2d tensors
assert_close([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [4.0, 4.0]]) # compare sequence of 1d tensors

np.array((1, 2)) == torch.tensor((1, 2))
np.array((1, 2)) == np.array((1, 2))
complex(2, 0) == 2
2. == 2
[1, 2] == torch.tensor((1, 2))
2. == torch.tensor((2.,))
2. == np.array((2.,))