import torch

import mmap
with open('testfile.pth', 'rb+') as f:
    mm = mmap.mmap(f.fileno(), 0)
    torch.load(mm)