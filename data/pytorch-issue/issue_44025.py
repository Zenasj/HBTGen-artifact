import random

py
import torch
import numpy as np
import time
import torch.multiprocessing as mp

def torch_cat(_, in_xs, out_x):
    s = time.perf_counter()
    torch.cat(in_xs)
    print("torch_cat", time.perf_counter() - s)


def torch_cat_out(_, in_xs, out_x):
    s = time.perf_counter()
    torch.cat(in_xs, out=out_x)
    print("torch_cat_out", time.perf_counter() - s)
    assert torch.eq(torch.cat(in_xs), out_x).all()


def torch_index_copy(_, in_xs, out_x):
    s_time = time.perf_counter()
    part_size = in_xs[0].shape[0]
    for i, in_x in enumerate(in_xs):
        idxs =  torch.arange(i * part_size, (i+1) * part_size)
        out_x.index_copy_(0, idxs, in_x)
    print("torch_index_copy", time.perf_counter() - s_time)
    assert torch.eq(torch.cat(in_xs), out_x).all()

def numpy_cat(_, in_xs, out_x):
    s = time.perf_counter()
    np.concatenate(in_xs)
    print("numpy_cat", time.perf_counter() - s)

if __name__ == '__main__':

    n_procs = 10
    n_parts = 8
    shape = (8, 128, 100)

    in_xs = [torch.randn(*shape) for _ in range(n_parts)]
    out_x = torch.randn(shape[0] * n_parts, shape[1], shape[2])
    for func in [torch_cat, torch_cat_out, torch_index_copy]:
        for i in range(5):
            mp.spawn(func, nprocs=n_procs, args=(in_xs, out_x), join=True)

    in_xs = [np.random.randn(*shape) for _ in range(n_parts)]
    out_x = np.random.randn(shape[0] * n_parts, shape[1], shape[2])
    for func in [numpy_cat]:
        for i in range(5):
            mp.spawn(func, nprocs=n_procs, args=(in_xs, out_x), join=True)