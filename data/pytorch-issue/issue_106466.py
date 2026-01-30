# foo.py
import torch
import time

min_sum = lambda a, b: torch.min(a, b).sum(-1)
min_sum_compiled = torch.compile(min_sum)

static_shapes = [
    ((192,), (192,)),
    ((96,), (96,))
]

# warmup
for k in range(100):
    for a_shape, b_shape in static_shapes:
        a = torch.rand(*a_shape, dtype = torch.float32)
        b = torch.rand(*b_shape, dtype = torch.float32)
        min_sum(a, b)
        min_sum_compiled(a, b)

for a_shape, b_shape in static_shapes:
    a = torch.rand(*a_shape, dtype = torch.float32)
    b = torch.rand(*b_shape, dtype = torch.float32)
    tic = time.time()
    for k in range(100):
        min_sum(a, b)
    print('min_sum', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')

    tic = time.time()
    for k in range(100):
        min_sum(a, b)
    print('min_sum_compiled', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')

import torch
import time

min_sum = lambda a, b: torch.min(a, b).sum(-1)
min_sum_compiled = torch.compile(min_sum)

torch._dynamo.reset()
min_sum_compiled_dynamic = torch.compile(min_sum, dynamic = True)

static_shapes = [
    ((192,), (192,)),
    ((96,), (96,))
]

dynamic_shapes = [
    ((6, 183, 1, 192), (6, 1, 183, 192)),
    ((6, 183, 1, 96), (6, 1, 183, 96))
]

for k in range(100):
    for a_shape, b_shape in (static_shapes + dynamic_shapes):
        a = torch.rand(*a_shape, dtype = torch.float32)
        b = torch.rand(*b_shape, dtype = torch.float32)
        min_sum(a, b)
        min_sum_compiled(a, b)
        min_sum_compiled_dynamic(a, b)

for a_shape, b_shape in (static_shapes + dynamic_shapes):
    a = torch.rand(*a_shape, dtype = torch.float32)
    b = torch.rand(*b_shape, dtype = torch.float32)
    tic = time.time()
    for k in range(100):
        min_sum(a, b)
    print('min_sum', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')

    tic = time.time()
    for k in range(100):
        min_sum_compiled(a, b)
    print('min_sum_compiled', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')

    tic = time.time()
    for k in range(100):
        min_sum_compiled_dynamic(a, b)
    print('min_sum_compiled_dynamic', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')

# foo.py
import torch
import time

# parse command line args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dynamic', action='store_true')
parser.add_argument('--simdlen', type=int, default=512)
args = parser.parse_args()

from torch._inductor import config
config.cpp.simdlen = args.simdlen
min_sum = lambda a, b: torch.min(a, b).sum(-1)
min_sum_compiled = torch.compile(min_sum, dynamic=args.dynamic)

static_shapes = [
    ((192,), (192,)),
    ((96,), (96,))
]

# warmup
for k in range(100):
    for a_shape, b_shape in static_shapes:
        a = torch.rand(*a_shape, dtype = torch.float32)
        b = torch.rand(*b_shape, dtype = torch.float32)
        min_sum(a, b)
        min_sum_compiled(a, b)

for a_shape, b_shape in static_shapes:
    a = torch.rand(*a_shape, dtype = torch.float32)
    b = torch.rand(*b_shape, dtype = torch.float32)
    tic = time.time()
    for k in range(100):
        min_sum(a, b)
    print('min_sum', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')

    tic = time.time()
    for k in range(100):
        min_sum(a, b)
    print('min_sum_compiled', a_shape, b_shape, (time.time() - tic) * 1000, 'ms')