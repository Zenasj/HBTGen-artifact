import torch._dynamo
from torch._inductor.utils import fresh_inductor_cache
import time

@torch._dynamo.optimize("inductor")
def foo():
    return torch.randn(10000)

def fresh_cache(fn):
    def inner(*args, **kwargs):
        cache_entries = {}
        cache_minder = fresh_inductor_cache(cache_entries)
        
        with cache_minder:
            return fn(*args, **kwargs)
    return inner

fresh_cache(foo)()

# warm up
start = time.time()
for _ in range(100):
    #foo()
    fresh_cache(foo)()
end = time.time()
print('average warmup: {:.2f} ms'.format((end-start)/100*1000))

# measure
start = time.time()
for _ in range(100):
    #foo()
    fresh_cache(foo)()
end = time.time()
print('average measure: {:.2f} ms'.format((end-start)/100*1000))

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU]) as prof:
    #foo()
    fresh_cache(foo)()
print(prof.key_averages().table(sort_by="self_cpu_time_total"))