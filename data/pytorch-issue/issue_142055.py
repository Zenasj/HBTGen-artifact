import torch

class Foo():
    def __init__(self, x):
        self.x = x

    def __eq__(self, other):
        return self.x == other.x

@torch.compile(fullgraph=True, backend="eager")
def run(x):
    f1 = Foo(0)
    f2 = Foo(0)
    return x + 1, f1 != f2

# prints (tensor([2.]), True)
# should be (tensor([2.]), False)
print(run(torch.ones(1)))