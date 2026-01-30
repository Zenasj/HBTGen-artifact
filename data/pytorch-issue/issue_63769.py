import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

import torch
from torch.nn.quantized import functional as qF

def verify(_s1, _zp1, _s2, _zp2, _conv_scale, _conv_zp):
    inputs = torch.rand(4, 4).reshape(1, 1, 4, 4)
    filters = torch.rand(4, 4).reshape(1, 1, 4, 4)
    
    s1, zp1 = _s1, _zp1  # scale and zero_point of the input activation
    s2, zp2 = _s2, _zp2  # scale and zero_point of the kernel

    q_inputs = torch.quantize_per_tensor(inputs, s1, zp1, dtype=torch.quint8)
    q_filters = torch.quantize_per_tensor(filters, s2, zp2, dtype=torch.qint8)

    conv_scale, conv_zp = _conv_scale, _conv_zp

    res = torch.tensor(0, dtype=torch.int)
    for i in range(filters.shape[2]):
        for j in range(filters.shape[3]):
            temp = (q_inputs.int_repr()[0, 0, i, j] - zp1) * (q_filters.int_repr()[0, 0, i, j] - zp2)
            res = res + temp  # multiply and accumulate into the res tensor

    res = conv_zp + (res * (s1 * s2 / conv_scale)).round()
    res = 255 if res > 255 else 0 if res < 0 else res  # handles scale and zero point

    if res != qF.conv2d(q_inputs, q_filters, bias=torch.tensor([0], dtype=torch.float), scale=conv_scale, zero_point=conv_zp).int_repr().item():
        print("!!!!!")
        print(_s1, _zp1, _s2, _zp2, _conv_scale, _conv_zp)
        print("myres: ", res, "qF.conv2d: ", qF.conv2d(q_inputs, q_filters, bias=torch.tensor([0], dtype=torch.float), scale=conv_scale, zero_point=conv_zp).int_repr()[0, 0, 0, 0].item())
        test_s1, test_zp1, test_s2, test_zp2, test_conv_scale, test_conv_zp = _s1, _zp1, _s2, _zp2, _conv_scale, _conv_zp
        raise Exception  # raises exception once a mismatch is detected

for i in range(100000):
    verify(_s1=np.random.rand(), _zp1=45, _s2=np.random.rand(), _zp2=0, _conv_scale=np.random.rand(), _conv_zp=0)