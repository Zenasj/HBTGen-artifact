import random

import numpy as np
import torch

n_images = 10 # need > 4 to see error

perms = np.random.permutation(n_images)
batch1 = np.random.randn(n_images,3,32,32)
batch2 = batch1[perms]
print(np.sum(np.abs(batch1))) # >>> 24666.88446814783
print(np.sum(np.abs(batch2))) # >>> 24666.88446814783 --- SAME

perms = torch.randperm(n_images)
batch1 = torch.empty(n_images,3,32,32).normal_(mean=0, std=1)
batch2 = batch1[perms]
print(torch.sum(torch.abs(batch1))) # >>> 24325.0156
print(torch.sum(torch.abs(batch2))) # >>> 24325.0117 --- NOT THE SAME