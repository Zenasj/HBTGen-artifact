import torch
from torch import multiprocessing as mp

FAIL = True

def f(a=1):
    torch.rand(3).requires_grad_(True).mean().backward()
    return a ** 2

if FAIL:
    f()

# This always works
p = mp.Process(target=f)
p.start()
p.join()

# This fails if autograd has been used
with mp.Pool(3) as pool:
    result = pool.map(f, [1, 2, 3])
print(result)

if __name__ == '__main__':
    mp.set_start_method("spawn")