py
import torch
from torch.profiler import profile, schedule

def f(warmup, active, repeat):
    with profile(schedule=schedule(skip_first=0, wait=0, warmup=warmup, active=active, repeat=repeat)) as prof:
        for _ in range(100):
            torch.add(1, 2)
            prof.step()

    print(f"{warmup=} {active=} {repeat=}")
    try:
        add = next(ev for ev in prof.key_averages() if ev.key == "aten::add")
        print(f"aten::add was called {add.count} times")
    except StopIteration:
        print(f"aten::add was called 0 times")
    print()

f(warmup=0, active=5, repeat=0)
f(warmup=0, active=5, repeat=1)
f(warmup=1, active=5, repeat=0)
f(warmup=1, active=5, repeat=1)
f(warmup=1, active=5, repeat=10)

f(warmup=0, active=10, repeat=0)
f(warmup=0, active=10, repeat=1)
f(warmup=1, active=10, repeat=0)
f(warmup=1, active=10, repeat=1)

f(warmup=0, active=10, repeat=10)
f(warmup=1, active=10, repeat=10)