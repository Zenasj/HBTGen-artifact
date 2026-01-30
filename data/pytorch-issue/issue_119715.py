import torch

class T:
    def __init__(self, x):
        self.x = x

@torch.compile()
def fn(x):
    o = T(5)
    return o, x + 1

fn(torch.ones(1, 1))
T = 5
fn(torch.ones(1, 1))