import random

import numpy as np
import torch
import logging


def test(foo, bar):
    return np.count_nonzero(foo) + np.count_nonzero(bar)

def test2(foo, bar):
    for clause in foo:
        for i in range(0, bar.shape[1]):
            res = np.logical_xor(clause, bar[i])
            if np.count_nonzero(res) == 1:
                bar[i] |= res
    return bar


foo = np.random.choice([True]*5 + [False]*5, size=(10,10))
bar = np.random.choice([True] + [False]*10, size=(10,10))

torch._logging.set_logs(all=logging.WARNING, output_code=True)

compiled = torch.compile(test2)
compiled(foo, bar)

import numpy as np
import torch
import logging


def test(foo, bar):
    return np.count_nonzero(foo) + np.count_nonzero(bar)

def test2(foo, bar):
    for clause in foo:
        for i in range(0, bar.shape[1]):
            res = np.logical_xor(clause, bar[i])
            if np.count_nonzero(res) == 1:
                bar[i] |= res
    return bar


foo = np.random.choice([True]*5 + [False]*5, size=(10,10))
bar = np.random.choice([True] + [False]*10, size=(10,10))

torch._logging.set_logs(all=logging.WARNING, output_code=True)

compiled = torch.compile(test2) # works for test
compiled(foo, bar)

def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

def f1(x, y):
    if x.sum() < 0:
        return -y
    return y

def test2(foo, bar):
    for clause in foo:
        for i in range(0, bar.shape[1]):
            res = np.logical_xor(clause, bar[i])
            if np.count_nonzero(res) == 1:
                bar[i] |= res
    return bar

def test2(foo, bar):
    for clause in foo:
        for i in range(0, bar.shape[1]):
            res = np.logical_xor(clause, bar[i])
            bar[i] |= np.logical_and(bool(np.count_nonzero(res)), res)
    return bar