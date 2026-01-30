import argparse
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio
import torchvision


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validation(args, pr, net, device='cuda'):
    import pdb; pdb.set_trace()
    batch_size = 1568
    net.eval()
    with torch.no_grad():
        for i in range(100):
            img = torch.rand(batch_size, 3, 64, 64).to(device)
            out = net(img)
    torch.cuda.empty_cache()
    return None

def train(args, pr, net, device='cuda'):
    import pdb; pdb.set_trace()
    net.train()
    batch_size = 1568
    for i in range(100):
        img = torch.rand(batch_size, 3, 64, 64).to(device)
        out = net(img)
    torch.cuda.empty_cache()
    return None

def main(args, device):
    # save dir
    gpus = torch.cuda.device_count()
    gpu_ids = list(range(gpus))

    # ----- Network ----- #
    net = torchvision.models.resnet18(pretrained=True, progress=True).to(device)
    net = nn.DataParallel(net, device_ids=gpu_ids)
    pr = None
    #  --------- Random or resume validation ------------ #
    res = validation(args, pr, net, device)
    res = train(args, pr, net, device)


if __name__ == '__main__':
    args = None
    main(args, DEVICE)