import random

import numpy as np
from multiprocessing import Process
import torch

def do_task(N):
    mat = np.random.randn(N ** 3)
    a = torch.from_numpy(mat)
    print("a")
    a.float()
    print("b")

N = 40  # hangs only if set larger (> 30)
do_task(N)  # don't hang without this line
p = Process(target=do_task, args=(N,))
p.start()
p.join()

3
import contextlib
import numpy as np
from multiprocessing import Process
import torch

@contextlib.contextmanager
def num_torch_thread(n_thread: int):
    n_thread_original = torch.get_num_threads()
    torch.set_num_threads(n_thread)
    yield
    torch.set_num_threads(n_thread_original)

def do_task(N):
    mat = np.random.randn(N ** 3)
    a = torch.from_numpy(mat)
    print("a")
    with num_torch_thread(1):
        a.float()
    print("b")

N = 100  # hangs only if set larger (> 30)
do_task(N)  # don't hang without this line
p = Process(target=do_task, args=(N,))
p.start()
p.join()