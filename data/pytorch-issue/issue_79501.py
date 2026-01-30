import torch
import torch.nn as nn

import numpy as np
import random

def set_seed(seed):
    """
    For seed to some modules.
    :param seed: int. The seed.
    :return:
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
set_seed(100)

grus = nn.GRU(512, 512, 24)
grus.eval()
input = torch.rand(32, 1, 512)


# method1：inference a batch
h0 = torch.zeros((24, 1, 512), dtype=input.dtype, device=input.device)
output, _ = grus(input, h0)

# method2: inference one by one for a batch
output_s = None
for t in range(32):
    h0 = torch.zeros((24, 1, 512), dtype=input.dtype, device=input.device)
    grus.eval()
    sub_out, _ = grus(input[[t], :, :], h0)
    if t == 0:
        output_s = sub_out
    else:
        output_s = torch.cat([output_s, sub_out], dim=0)

for t in range(32):
    print(f"{t}th frame: {torch.allclose(output[[t], :, :], output_s[[t], :, :])}", end="")
    print(f"\t{torch.mean(output[[t], :, :] - output_s[[t], :, :])}")