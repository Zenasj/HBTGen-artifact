import torch
from test_jit import get_execution_plan
from common_utils import enable_profiling_mode
import sys
import time



def abs100(x):
    for i in range(100000):
        x = x.abs()
    return x

def abs1(x):
    return x.abs()

with enable_profiling_mode():
    x = torch.ones(1)
    ja = torch.jit.trace(abs100, (x,), None, False, False)
    ja(x)
    ja(x)

    # ja = torch.jit.script(abs1)
    # ja(x)
    # ja(x)

    start = time.perf_counter_ns()
    for i in range(1):
        ja(x)
    #duration = time.process_time_ns() - start
    duration = (time.perf_counter_ns() - start) / 1000000000
    print("time = {}".format(duration))