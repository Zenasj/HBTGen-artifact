import random
import torch

def fn(x):
    seed = random.randint(0, 100)
    rand = random.Random(seed)
    return x + rand.randrange(10)

opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
opt_fn(torch.ones(1))