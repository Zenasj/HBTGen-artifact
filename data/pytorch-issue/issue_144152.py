from multiprocessing import Process

import torch

torch.device(0) # Note that torch.device('cuda') or torch.device('cuda:0') do not trigger the issue

def cuda_init():
    torch.Tensor([0]).cuda()

p = Process(target=cuda_init)
p.start()
p.join()
assert p.exitcode == 0