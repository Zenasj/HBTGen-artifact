import torch
import torch.multiprocessing as mp

import pyro
import pyro.distributions as dist

torch.set_default_tensor_type(torch.cuda.FloatTensor)
n = 10

def model():
    loc = pyro.sample("loc", dist.Normal(torch.zeros(3), 1))
    pyro.sample("y", dist.Normal(loc, 1))  # comment out this line -> no error

def worker(q, e):
    for i in range(n):
        trace = {"normal": dist.Normal(0, 1)}  # comment out this line -> no error ???
        trace = pyro.poutine.trace(model).get_trace()  # this is just a dictionary which holds tensors
        q.put(trace.nodes["loc"])  # q.put(trace) also gives error
        e.wait()
        e.clear()

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    e = ctx.Event()
    p = ctx.Process(target=worker, args=(q, e))
    p.start()
    for i in range(n):
        trace = q.get()
        e.set()
    p.join()