import os
import torch
import multiprocessing as mp

def test(a):
    os.sched_setaffinity(0, [2])
    torch.randperm(1)
    return os.sched_getaffinity(0)

if __name__ == '__main__':
    mp.set_start_method('spawn') # or 'forkserver'
    with mp.Pool(2) as p:
        print(p.map(test, range(2)))