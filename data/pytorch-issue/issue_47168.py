# test.py

import torch

for i in range(1000):

    gpu_arr = torch.randn(size=[2, 800]).cuda()

    success = (gpu_arr[1].sum() == gpu_arr.sum(dim=1)[1])

    if not success:
        print("Test failed!")
        exit()

print("Test passed!")