import torchvision

py
def f_as_tensor(x):
    mean = torch.as_tensor((1,2,3,6,7,8,9,10), dtype=x.dtype, device=x.device)
    return mean

py
def f_as_tensor(x):
    """
    leaks ~32bytes
    """
    mean = torch.as_tensor((1,2,3,6,7,8,9,10), dtype=x.dtype, device=x.device)
    return mean

def f_mul(x):
    """
    leaks ~64bytes
    """
    return x * 10

def f_view(x):
    """
    no leak
    """
    return x.view(-1)
    
# leaks ~11kb/it
m = torchvision.models.resnet18()

py
import torch
from torchvision import models
import tracemalloc
import gc

device = torch.device('cuda')
m = models.resnet18().to(device)
m = torch.compile(m)
inp = torch.rand(2, 3, 240, 320, device=device)

# warmup
gc.collect()
tracemalloc.start()
for i in range(10):
    m(inp)
    
gc.collect()
    
print('warmup done')
snapshot1 = tracemalloc.take_snapshot()

start_mem = get_mem()
for i in range(2000):
    m(inp)
end_mem = get_mem()
print('consumed', end_mem-start_mem)

snapshot2 = tracemalloc.take_snapshot()
top_stats = snapshot2.compare_to(snapshot1, 'lineno')


print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)