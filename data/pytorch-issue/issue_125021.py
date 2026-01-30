import torch

@torch.compile(backend="eager")
def f(x):
    with torch.profiler.profile() as p:
        pass
    p.profiler.kineto_results.experimental_event_tree()
    return x + 1

x = torch.ones(1)
f(x)