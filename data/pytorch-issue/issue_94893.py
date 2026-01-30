import torch
import random
import time
import subprocess

iters = 100

last_large_dim = -1
device = torch.device('cuda:2')

for i in range(iters):
    # removing the random size of this dimension removes the increase in memory on each iteration
    rand_dim = random.randint(8000, 15000)
    x = torch.randn(1, 768, rand_dim).to(device)

    # removing this line prevents memory from increasing 
    res = torch.fft.rfft(x, n=rand_dim)
    
    # even detaching and deleting does not clear the memory
    x = x.detach().cpu()
    del x
    
    res = res.detach().cpu()
    del res
    
    # regardless of whether we pass the largest dimension this happens
    if rand_dim > last_large_dim:
        last_large_dim = rand_dim
        print ('Newest largest dim created')
        
    # report from nvidia-smi
    torch.cuda.empty_cache()
    subprocess.run('nvidia-smi -g 2', shell=True)
    
    time.sleep(0.5)

torch.backends.cuda.cufft_plan_cache[2].max_size = 32

import torch
import random
import time
import subprocess

iters = 1000000
uniques = set()
count = 0
maxmem = 0
maxcount = 0
maxuniques = 0
maxiter = 0

last_large_dim = -1
device = torch.device('cuda:0')

torch.backends.cuda.cufft_plan_cache[0].max_size = 0

for i in range(iters):
    # removing the random size of this dimension removes the increase in memory on each iteration
    rand_dim = random.randint(8000, 15000)
    x = torch.randn(1, 768, rand_dim).to(device)
    uniques.add(rand_dim)

    # removing this line prevents memory from increasing
    res = torch.fft.rfft(x, n=rand_dim)

    # even detaching and deleting does not clear the memory
    x = x.detach().cpu()
    del x

    res = res.detach().cpu()
    del res

    # regardless of whether we pass the largest dimension this happens
    if rand_dim > last_large_dim:
        last_large_dim = rand_dim
        print ('Newest largest dim created', count)


    # report from nvidia-smi
    torch.cuda.empty_cache()
    out = subprocess.check_output('nvidia-smi --query-gpu=memory.used --format=csv', shell=True)
    mb = int(out.decode('ascii').splitlines()[1].split(' ')[0])
    if mb > maxmem:
        count += 1
        maxmem = mb
        maxcount = count
        maxiter = i
        maxuniques = len(uniques)

    print(f"{mb} @ ({torch.backends.cuda.cufft_plan_cache.size}) iter {i} {count} {len(uniques)} prev max at {maxiter} {maxcount} {maxuniques}")