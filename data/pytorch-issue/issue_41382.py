import torch.nn as nn

import torch
x = torch.randn(1, 64, 40000)
if torch.cuda.is_available():
    x = x.cuda()
trans_conv = torch.nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
if torch.cuda.is_available():
    trans_conv.to('cuda')
    
import time
num = 100
with torch.no_grad():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for i in range(num):
        y = trans_conv(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    print('average cost: {}ms'.format((end - start) * 1000 / num))