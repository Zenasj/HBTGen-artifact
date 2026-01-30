import torch.multiprocessing as mp
import time


def proc_target(x):
    time.sleep(100000)      


if __name__ == '__main__':
	context = mp.spawn(fn=proc_target, nprocs=32)

import multiprocessing as mp
import time

def proc_target():
    time.sleep(100000)      

if __name__ == '__main__':
    mp.set_start_method('spawn')
    for i in range(32): 
        p = mp.Process(target=proc_target)
        p.start()