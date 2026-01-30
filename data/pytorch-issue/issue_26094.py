import torch
import GPUtil
import gc


def gpu_test():
    x = torch.ones([2, 4], dtype=torch.float64, device=torch.device('cuda:0'))
    print('After init on gpu 0')
    GPUtil.showUtilization()
    for device in ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu']:
        x = x.to(device)
        print("Tensor on device:", x.get_device())
        GPUtil.showUtilization()
        torch.cuda.empty_cache()
        print("After torch.cuda.empty_cache()")
        GPUtil.showUtilization()


print("Startup:")
GPUtil.showUtilization()
gpu_test()
print("All tensors out of function scope")
GPUtil.showUtilization()
print("After torch.cuda.empty_cache()")
GPUtil.showUtilization()
print("After gc.collect()")
del torch
gc.collect()
GPUtil.showUtilization()

import torch.multiprocessing as _mp
import torch
import time
import GPUtil

mp = _mp.get_context('spawn')


class Process(mp.Process):
    def __init__(self):
        super().__init__()
        print("Init Process")
        return

    def run(self):
        x = torch.ones([2, 4], dtype=torch.float64, device=torch.device('cuda:0'))
        for device in ['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu']:
            x = x.to(device)
            torch.cuda.empty_cache()

        del x
        torch.cuda.empty_cache()
        time.sleep(15)


if __name__ == "__main__":
    num_processes = 12
    print("Example using", num_processes, "processes")
    print("Startup:")
    GPUtil.showUtilization()
    processes = [Process() for i in range(num_processes)]
    print("After Process creation")
    GPUtil.showUtilization()
    [p.start() for p in processes]
    print("Directly after process start")
    GPUtil.showUtilization()
    time.sleep(5)
    print("After 5 seconds")
    GPUtil.showUtilization()
    time.sleep(5)
    print("After 10 seconds")
    GPUtil.showUtilization()
    time.sleep(5)
    print("After 15 seconds and empty cache")
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    time.sleep(5)
    print("After 20 seconds and empty cache")
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    time.sleep(5)
    print("After 25 seconds and empty cache")
    torch.cuda.empty_cache()
    GPUtil.showUtilization()
    [p.join() for p in processes]
    torch.cuda.empty_cache()
    print("After process join and empty cache")
    GPUtil.showUtilization()

import torch.multiprocessing as _mp
import torch
import os

mp = _mp.get_context('spawn')


class Process(mp.Process):
    def __init__(self):
        super().__init__()
        print("Init Process")
        return

    def run(self):
        print("Hello World!")
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        print(torch.cuda.device_count())
        print(torch.cuda.current_device())

if __name__ == "__main__":
    num_processes = 2
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    processes = [Process() for i in range(num_processes)]
    [p.start() for p in processes]
    print("main: " + os.environ['CUDA_VISIBLE_DEVICES'])
    [p.join() for p in processes]