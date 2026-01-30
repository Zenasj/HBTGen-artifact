import numpy as np

a = np.array([[56.6896,  4.7862, 10.0108], [ 4.7059, 90.4238, 10.5659], [10.3995, 11.0197, 12.6870]])

import torch

b = torch.from_numpy(a).cuda()


print(torch.linalg.svd(b))

print(torch.linalg.svd(b))