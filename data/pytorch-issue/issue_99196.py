import torch.nn as nn

#!/usr/local/bin/python

import subprocess
import torch

filename = 'torch_nn_linear.pt'
batch_size = 32
bias_flag = False

# This is from https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def trace():
    model = torch.nn.Linear(in_features=256, out_features=128, bias=bias_flag).cuda()

    input = torch.Tensor(32, 8, 256).cuda()
    for j in range(100):
        y = model(input)

    traced = torch.jit.trace(model, input)

    print(traced.code)
    traced.save(filename)
    print(f'Saved to {filename}')


def my_func():
    model = torch.jit.load(filename).cuda()
    print(f"Loaded {filename}")

    static_input = torch.randn(batch_size, 8, 256).cuda()

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(30):
            pred = model(static_input)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pred = model(static_input)
        gpu_mem = get_gpu_memory_map()
        print("After capture GPU: {:.2f} KB".format(gpu_mem[0]))

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            print("Start replaying")

            for i in range(100):
                g.replay()
                if i % 10 == 0:
                    gpu_mem = get_gpu_memory_map()
                    print("GPU: {:.2f} KB".format(gpu_mem[0]))

trace()
print(f'Run {filename} with Cuda graph. Batch size {batch_size}')
for i in range(10):
    my_func()
print(f'Finished running {filename} with Cuda graph. Batch size {batch_size}')

#!/usr/local/bin/python

import subprocess
import torch

filename = 'torch_matmul_linear.pt'
batch_size = 32

class TorchMatmulLayer(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(input=x, other=y)

# This is from https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def trace():
    model = TorchMatmulLayer().cuda()

    x = torch.randn(32, 256).cuda()
    y = torch.Tensor(256, 256).cuda()
    
    for i in range(100):
        pred = model(x, y)

    traced = torch.jit.trace(model, (x, y))

    print(traced.code)
    traced.save(filename)
    print(f'Saved to {filename}')


def my_func():
    model = torch.jit.load(filename).cuda()
    print(f"Loaded {filename}")

    x = torch.randn(batch_size, 256).cuda()
    y = torch.Tensor(256, 256).cuda()

    # warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(30):
            pred = model(x, y)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pred = model(x, y)
        gpu_mem = get_gpu_memory_map()
        print("After capture GPU: {:.2f} KB".format(gpu_mem[0]))

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            print("Start replaying")

            for i in range(100):
                g.replay()
                if i % 10 == 0:
                    gpu_mem = get_gpu_memory_map()
                    print("GPU: {:.2f} KB".format(gpu_mem[0]))

trace()
print(f'Run {filename} with Cuda graph. Batch size {batch_size}')
for i in range(10):
    my_func()
print(f'Finished running {filename} with Cuda graph. Batch size {batch_size}')

for i in range(10):
    my_func()
    torch.cuda.empty_cache()

#!/usr/local/bin/python

import subprocess
import torch

filename = 'torch_nn_linear.pt'
batch_size = 32
bias_flag = False

# This is from https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def trace():
    model = torch.nn.Linear(in_features=256, out_features=128, bias=bias_flag).cuda()

    input = torch.Tensor(32, 8, 256).cuda()
    for j in range(100):
        y = model(input)

    traced = torch.jit.trace(model, input)

    print(traced.code)
    traced.save(filename)
    print(f'Saved to {filename}')


def my_func(s):
    model = torch.jit.load(filename).cuda()
    print(f"Loaded {filename}")

    static_input = torch.randn(batch_size, 8, 256).cuda()

    # warmup
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(30):
            pred = model(static_input)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pred = model(static_input)
        gpu_mem = get_gpu_memory_map()
        print("After capture GPU: {:.2f} KB".format(gpu_mem[0]))

    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            print("Start replaying")

            for i in range(100):
                g.replay()
                if i % 10 == 0:
                    gpu_mem = get_gpu_memory_map()
                    print("GPU: {:.2f} KB".format(gpu_mem[0]))

trace()
print(f'Run {filename} with Cuda graph. Batch size {batch_size}')
s = torch.cuda.Stream()
for i in range(10):
    my_func(s)
print(f'Finished running {filename} with Cuda graph. Batch size {batch_size}')