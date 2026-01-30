import torch
import torch.multiprocessing as mp
import time

def producer(queue):
    while True:
        a = torch.ones(2,2).float().cuda()
        idx = torch.LongTensor([[0, 0], [0, 1]]).cuda()
        queue.put((a, idx))

def consumer(queue):
    while True:
        a, idx = queue.get()
        print(idx.type())
        print(idx)

if __name__ == '__main__':
    mp.set_start_method('spawn')

    queue = mp.Queue()

    p = mp.Process(target=producer, args=(queue,))
    c = mp.Process(target=consumer, args=(queue,))
    p.start()
    c.start()
    
    time.sleep(10)

    p.join()
    c.join()

import torch
import torch.multiprocessing as mp
import time

def producer(queue, event):
    while True:
        for _ in range(10):
            a = torch.ones(2,2).float().cuda()
            idx = torch.ByteTensor([[0, 0], [0, 1]]).cuda()

            queue.put(a)
            queue.put(idx)
            event.wait()
            event.clear()
        time.sleep(1000)
        return

def consumer(queue, event):
    while True:
        for _ in range(10):
            idx = queue.get()
            temp = queue.get()
            print("CONSUMER ", idx)
            print("CONSUMER ", temp)
            del idx
            del temp
            event.set()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    queue = mp.Queue()
    queue2 = mp.Queue()

    event = mp.Event()
    p = mp.Process(target=producer, args=(queue, event))
    c = mp.Process(target=consumer, args=(queue, event))
    p.start()
    c.start()

    time.sleep(10)

    p.join()
    c.join()