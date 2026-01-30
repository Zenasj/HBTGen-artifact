# test.py
import os
import torch
import time
import sys

print(os.environ)
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(os.getpid())
sys.stdout.flush()

device = torch.device('cuda')
a = torch.randn(10, 10, device=device)

os.system('nvidia-smi')