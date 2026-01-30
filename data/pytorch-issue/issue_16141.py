import random

import torch
import torch.distributions as dist
import torch.multiprocessing as mp

torch.set_default_tensor_type(torch.cuda.FloatTensor)
n = 10

def model():
    d1 = dist.Normal(torch.zeros(3), 1)
    v1 = d1.rsample()  # no error when replace `.rsample()` by `.sample()`
    d2 = dist.Normal(v1, 2)
    v2 = d2.rsample()
    return [(d1, v1), (d2, v2)]

def worker(q, e):
    for i in range(n):
        sample = [torch.zeros(1), torch.ones(1)]  # no error when remove this line ???
        sample = model()
        q.put(sample)
        e.wait()
        e.clear()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    e = ctx.Event()
    p = ctx.Process(target=worker, args=(q, e))
    p.start()
    for i in range(n):
        print("=== ITER {} ===".format(i))
        sample = q.get()
        print(sample)
        e.set()
    p.join()

import torch
import torch.distributions as dist
import torch.multiprocessing as mp

torch.set_default_tensor_type(torch.cuda.FloatTensor)
n = 10

def model():
    d1 = dist.Normal(torch.zeros(3), 1)
    v1 = d1.rsample()  # no error when replace `.rsample()` by `.sample()`
    d2 = dist.Normal(v1, 2)
    v2 = d2.rsample()
    return [({"dist": d1, "value": v1}), ({"dist": d2, "value": v2})]

def worker(q, e):
    for i in range(n):
        sample = [torch.zeros(1), torch.ones(1)]  # no error when remove this line ???
        sample = model()
        q.put(sample)
        e.wait()
        e.clear()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    e = ctx.Event()
    p = ctx.Process(target=worker, args=(q, e))
    p.start()
    samples = []
    for i in range(n):
        print("=== ITER {} ===".format(i))
        sample = q.get()
        #del sample
        e.set()
    p.join()

import torch
import torch.distributions as dist
import torch.multiprocessing as mp

torch.set_default_tensor_type(torch.cuda.DoubleTensor)
n = 10

def model():
    return torch.rand(3)

def worker(q, e):
    for i in range(n):
        sample = model()
        q.put(sample)
        e.wait()
        e.clear()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    e = ctx.Event()
    p = ctx.Process(target=worker, args=(q, e), daemon=True)
    p.start()
    samples = []
    real_examples = []
    for i in range(n):
        print("=== ITER {} ===".format(i))
        sample = q.get()
        real_examples.append(sample.clone())
        samples.append(sample)
        e.set()
    p.join()
    print(real_examples)
    print(samples)

import torch
import torch.distributions as dist
import torch.multiprocessing as mp

torch.set_default_tensor_type(torch.cuda.FloatTensor)
n = 10

def model():
    d1 = dist.Normal(torch.zeros(3), 1)
    v1 = d1.rsample()  # no error when replace `.rsample()` by `.sample()`
    d2 = dist.Normal(v1, 2)
    v2 = d2.rsample()
    return [({"dist": d1, "value": v1}), ({"dist": d2, "value": v2})]

def worker(q, e):
    for i in range(n):
        sample = [torch.zeros(1), torch.ones(1)]  # no error when remove this line ???
        sample = model()
        q.put(sample)
        e.wait()
        e.clear()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    e = ctx.Event()
    p = ctx.Process(target=worker, args=(q, e))
    p.start()
    samples = []
    for i in range(n):
        print("=== ITER {} ===".format(i))
        sample = q.get()
        del sample ### REQUIRED! Once you call e.set(), this memory is no longer valid. It can be random.
        e.set()
    p.join()