import torch
from torchvision import models
import time

net = models.resnet101(pretrained=True).cpu()
# artificially create a dilation layer 
net.conv1.dilation=(3, 3)
net = net.eval()

# 4batch size
example = torch.rand(4, 3, 224, 224).cpu()

with torch.no_grad():
    traced_script_module = torch.jit.trace(net, example)

traced_script_module.save('traced_resnet101_dilation.pt')

from __future__ import print_function

import os
import sys
import time
import torch

traced_script_module = torch.jit.load(sys.argv[1])
example = torch.rand(4, 3, 224, 224).cpu()

with torch.no_grad():
    for _ in range(5):
        start_ts = time.time()
        with torch.autograd.profiler.profile() as profile:
            traced_script_module(example)
        end_ts = time.time()
        print('inference took {:.2f}ms'.format((end_ts - start_ts) * 1000))

        profile.export_chrome_trace('profile_python_cpu.json')