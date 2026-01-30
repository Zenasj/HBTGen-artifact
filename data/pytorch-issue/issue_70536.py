import torch
from torch.multiprocessing import Queue, Process

def simple_agent(queue):
    for i in range(10):
        c = torch.Tensor([i]) # passing tensor will lead to error
        #c = i                # constant is OK
        queue.put(c)

def proc():
    queue = Queue(20)
    subp = Process(target=simple_agent, args=(queue,))
    subp.start()
    time.sleep(1) # make sure subprocess has exited
    # if do not sleep, result will be OK
    for i in range(10):
        v = queue.get()
        print(i, v)