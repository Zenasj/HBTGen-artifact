import torch.nn as nn
import random

import time
import numpy as np
import torch
from termcolor import colored
def time_avg_pool2d(X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override, iterations):
    X, (scale, zero_point, torch_type) = X
    qX_nchw = torch.quantize_per_tensor(torch.from_numpy(X), scale=scale,
                                    zero_point=zero_point, dtype=torch_type)
    qX_nhwc = qX_nchw.contiguous(memory_format=torch.channels_last)
    assert(qX_nhwc.stride() != sorted(qX_nhwc.stride()))
    assert(qX_nchw.is_contiguous(memory_format=torch.contiguous_format))
    assert(qX_nhwc.is_contiguous(memory_format=torch.channels_last))
    start = time.time()
    for _ in range(iterations):
        X_hat = torch.nn.quantized.functional.avg_pool2d(qX_nchw, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override)
    qnchw_end = time.time() - start
    start = time.time()
    for _ in range(iterations):
        X_hat = torch.nn.quantized.functional.avg_pool2d(qX_nhwc, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override)
    qnhwc_end = time.time() - start
    return qnchw_end*1000/iterations, qnhwc_end*1000/iterations

def time_avg_pool3d(X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override,  iterations):
    X, (scale, zero_point, torch_type) = X
    qX_ncdhw = torch.quantize_per_tensor(torch.from_numpy(X), scale=scale,
                                    zero_point=zero_point, dtype=torch_type)
    qX_ndhwc = qX_ncdhw.contiguous(memory_format=torch.channels_last_3d)
    assert(qX_ndhwc.stride() != sorted(qX_ndhwc.stride()))
    assert(qX_ncdhw.is_contiguous(memory_format=torch.contiguous_format))
    assert(qX_ndhwc.is_contiguous(memory_format=torch.channels_last_3d))
    start = time.time()
    for _ in range(iterations):
        X_hat = torch.nn.quantized.functional.avg_pool3d(qX_ncdhw, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override)
    qncdhw_end = time.time() - start
    start = time.time()
    for _ in range(iterations):
        X_hat = torch.nn.quantized.functional.avg_pool3d(qX_ndhwc, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                count_include_pad=count_include_pad, divisor_override=divisor_override)
    qndhwc_end = time.time() - start
    return qncdhw_end*1000/iterations, qndhwc_end*1000/iterations

iterations = 10000
print("iterations = {}".format(iterations))
print("Benchmark", "Time(ms)", sep="\t\t\t\t\t")
for torch_type in (torch.qint8, torch.quint8, torch.qint32):
    for channel in (4,8,64,256):
        X = np.random.rand(1, channel, 56, 56).astype(np.float32), (0.5, 1, torch_type)
        ts = time_avg_pool2d(X, 4, None, 0, True, True, None, iterations)
        print(colored("avg_pool2d({}, {}, {})".format(str(torch_type), channel, "nchw"), 'green'), colored(ts[0], 'yellow'), sep="\t")
        print(colored("avg_pool2d({}, {}, {})".format(str(torch_type), channel, "nhwc"), 'green'), colored(ts[1], 'yellow'), sep="\t")
for torch_type in (torch.qint8, torch.quint8, torch.qint32):
    for channel in (4,8,64,256):
        X = np.random.rand(1, channel, 56, 56, 4).astype(np.float32), (0.5, 1, torch_type)
        ts = time_avg_pool3d(X, 4, None, 0, True, True, None, iterations)
        print(colored("avg_pool3d({}, {}, {})".format(str(torch_type), channel, "ncdhw"), 'green'), colored(ts[0], 'yellow'), sep="\t")
        print(colored("avg_pool3d({}, {}, {})".format(str(torch_type), channel, "ndhwc"), 'green'), colored(ts[1], 'yellow'), sep="\t")