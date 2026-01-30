Python
import torch
import torch.multiprocessing as mp
import time

def func_return_list(queue):
    queue.put([[0,0],[0,0]])

def func(queue):
    queue.put( torch.zeros((2,2)) )
    time.sleep(1)

def func_with_bug(queue):
    queue.put( torch.zeros((2,2)) )

queue = mp.Queue()

# a function that puts a python list
p1 = mp.Process(target=func_return_list, args=(queue, ))
p1.start()
print(queue.get())
p1.join()

# a function that puts a pytorch tensor and sleeps for 1 sec
p2 = mp.Process(target=func, args=(queue, ))
p2.start()
print(queue.get())
p2.join()

# a function that puts a pytorch tensor
p3 = mp.Process(target=func_with_bug, args=(queue, ))
p3.start()
print(queue.get())
p3.join()