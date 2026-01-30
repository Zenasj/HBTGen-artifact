import random

import sys
import numpy as np
import torch
from torch import tensor
from torch.nn import MaxPool2d

DEVICE = "cpu" # gpu
FORMAT = "NCHW" # NHWC
DTYPE = "float32"
DELTA = 10 # [100, 50, 10, 1, 0.1, 0.01, 0.001]
SHAPE = (10, 3, 32, 32)
DATA_RANGE = [-1, 1]

def norm_dis(source_result, follow_result, ord="l1"):
    if ord == "l1":
        dis = torch.sum(torch.abs_(source_result-follow_result))
    elif ord == "l2":
        dis = torch.norm(source_result - follow_result)
    return dis

def generate_data(shape, device):
    return torch.from_numpy(np.random.uniform(DATA_RANGE[0], DATA_RANGE[1], shape).astype(DTYPE)).float().to(device)

def SourceModel():
    pool_size = np.random.randint(1, 5)
    strides = np.random.randint(1, 5)
    pool = MaxPool2d(kernel_size=(pool_size, pool_size), stride=(strides, strides))
    return pool


def maintest():
    torch.set_printoptions(precision=8)

    device = torch.device('cuda' if DEVICE == "gpu" else "cpu")
    shape = SHAPE
    d_format = torch.channels_last if FORMAT == "NHWC" else torch.contiguous_format
    for i in range(100):
        source_model = SourceModel().to(device).to(memory_format=d_format).eval()
        data = generate_data(shape, device).to(memory_format=d_format)

        delta = tensor(np.random.uniform(-DELTA, DELTA, 1)[0]).to(device)
        source_result = source_model.forward(data) * delta
        follow_result = source_model.forward(data * delta)
        dis = norm_dis(source_result, follow_result).cpu().detach().numpy()[()]
        if dis >= 1:
            print(dis)


if __name__ == "__main__":
    maintest()
    print("end")