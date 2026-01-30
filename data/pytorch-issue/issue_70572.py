import torch
from time import time

def fun(a):
    b = a==0
    return b & b

a = torch.zeros(3,device='cuda')
fun_trace = torch.jit.trace(fun,a)

for _ in range(4):
    for n in range(1,6):
        torch.cuda.synchronize()
        t = time()
        fun_trace(torch.zeros(n,device='cuda'))
        torch.cuda.synchronize()
        print(f'{n=} {int((time()-t)*1e3)} ms')