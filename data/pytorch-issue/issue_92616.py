import random

import numpy as np
import torch
import copy
from torch.nn import Conv2d
import os

def SourceModel(shape, device):
    model = Conv2d(in_channels=shape, out_channels=32, kernel_size=(5, 5)).to(device)
    return model

def FollowModel_4(source_model):
    # copy model
    follow_model = copy.deepcopy(source_model)
    # mutation model
    follow_model.train()
    follow_model.bias.data += delta
    follow_model.eval()
    return follow_model

DEVICE = "gpu"
if DEVICE == "gpu":
    # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if DEVICE == "gpu" else "cpu")
delta = 10
shape = (10, 3, 32, 32)
data = torch.from_numpy(np.random.uniform(-1, 1, shape)).float().to(device)

source_model = SourceModel(shape[1], device).to(device)
follow_model = FollowModel_4(source_model)
source_result = source_model.forward(data) + delta
follow_result = follow_model.forward(data)
dis = torch.sum(torch.abs_(source_result-follow_result)).cpu().detach().numpy()[()]

print("device:", DEVICE, "; delta:", delta, "; dis:", dis)