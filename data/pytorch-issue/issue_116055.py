import torch
import queue
import torch.multiprocessing as mp

class MyProcess(mp.Process):
    def __init__(self):
        super().__init__()
        self.queue = mp.Queue(200)

    def run(self):
        while 1:
            print('running')
            try:
                imgs = self.queue.get(timeout=1)
                print("queue.get() ended, got", imgs.mean())

                del imgs # save memory
            except queue.Empty:
                print("queue.Empty")
                continue

    def write_frames(self, a):
        self.queue.put(a)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_num_threads(1)

    proc = MyProcess()
    for i in range(10):
        a = torch.rand([8,1,320,320], device='cuda') + i
        proc.write_frames(a)
        del a # save memory
    proc.start()
    proc.join()

import torch
import queue
import torch.multiprocessing as mp

class MyProcess(mp.Process):
    def __init__(self):
        super().__init__()
        self.queue = mp.Queue(200)

    def run(self):
        while 1:
            print('running')
            try:
                imgs = self.queue.get(timeout=1).cuda()
                print("queue.get() ended, got", imgs.mean())

                del imgs # save memory
            except queue.Empty:
                print("queue.Empty")
                continue

    def write_frames(self, a):
        self.queue.put(a)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_num_threads(1)

    proc = MyProcess()
    for i in range(10):
        a = torch.rand([8,1,320,320]) + i
        proc.write_frames(a)
        del a # save memory
    proc.start()
    proc.join()