#!/usr/bin/env python
# encoding: utf-8

import os
import psutil
import sys
import torch
import torch.nn as nn

print(torch.__version__)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 64),
        )
        if sys.argv[1] == "orth":
            print("orth init")
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
        else:
            print("xavier init")
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)


if __name__ == "__main__":
    print(psutil.Process(os.getpid()).memory_info())
    model = Model()
    print(psutil.Process(os.getpid()).memory_info())