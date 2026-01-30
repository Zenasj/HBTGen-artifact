import torch
import time

a = torch.randn(4,4).cuda()
torch.cuda.synchronize()

num_iter = 100
def test1():
    s = time.time()
    for i in range(num_iter):
        x = torch.inverse(a)
    torch.cuda.synchronize()
    e = time.time()
    gpu_time = ((e - s)/num_iter)
    return  gpu_time

def test2():
    s = time.time()
    for i in range(num_iter):
        b = a.to('cpu')
        d = torch.inverse(b)
        y = d.to('cuda')
    torch.cuda.synchronize()
    e = time.time()
    cpu_time = ((e - s)/num_iter)
    return  cpu_time

gpu_time = test1()
print("GPU Time: {:.5f} ms".format(gpu_time*1000))

cpu_time = test2()
print("CPU Time: {:.5f} ms".format( cpu_time*1000))

print("cpu/gpu: {:.2f}x ".format(gpu_time/cpu_time))

py
x = torch.randn(4, 4, device='cuda', dtype=torch.float)
with torch.autograd.profiler.emit_nvtx():
    y = torch.inverse(x)

a = torch.randn(4000,4000).cuda()
torch.cuda.synchronize()
num_iter = 1