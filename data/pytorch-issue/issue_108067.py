import torch

@torch.compile(dynamic=True)
def fn1(a):
    return a + torch.zeros((3,))


@torch.compile(dynamic=True)
def fn2(a):
    return torch.full((3,), a)

fn1(5)
fn2(5)