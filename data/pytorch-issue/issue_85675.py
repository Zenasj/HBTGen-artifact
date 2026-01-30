import random

3
import torch
import numpy as np

def rotate_subset(data):
    return torch.concat([data[:, :2], torch.rot90(data[:, 2:])])

data = np.random.uniform(size=[2, 4])
cpu_result = rotate_subset(torch.as_tensor(data).float().cpu())
mps_result = rotate_subset(torch.as_tensor(data).float().to("mps")).cpu()
np.testing.assert_almost_equal(mps_result.numpy(), cpu_result.numpy())  # raises AssertionError

3

def rotate_subset(data):
    return torch.concat([data[:, :2], torch.rot90(data[:, 2:]).contiguous()])