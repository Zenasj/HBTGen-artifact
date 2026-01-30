import torch

@torch.compile(backend="eager")
def run(x, int_set, ns):
    int_set.update(map(int, ns))
    return x + 1
int_set = set()
ns = [1, 2, 1]
run(torch.ones(1), int_set, ns)