import random

import torch
import numpy as np

def torch_indexing_test(device):
    a = torch.arange(10000).float().to(device)
    b = torch.tensor(np.random.choice(100, 10000)).to(device).long()
    
    c1 = torch.zeros(100).to(device)
    c2 = torch.zeros(100).to(device)

    c1[b] += a
    c2[b] += a

    print(torch.max(c1-c2)) # should be zero

if __name__ == '__main__':
    torch_indexing_test(device="cpu")
    torch_indexing_test(device="cuda:0")

    # print output:
    # tensor(0.)
    # tensor(9307., device='cuda:0')