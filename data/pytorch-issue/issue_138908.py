import torch

@torch.compile(backend="inductor", dynamic=True)
def test1_fn(x, y):
    x = x + 10
    y[x] += y[x]

def test_1():
    x = torch.randint(-10, -9, (1 ,2),  dtype=torch.int64)
    y = torch.randn((2, 32), dtype=torch.float32)
    x_clone = x.clone()
    y_clone = y.clone()
    print(x_clone)
    print(y_clone)
    with torch.no_grad():
        x_clone = x_clone + 10
        print(x_clone)
        y_clone[x_clone] += y_clone[x_clone]
        print(y_clone)
        print(y)
        test1_fn(x, y)

        print(y_clone)
        print(y)
        assert torch.allclose(y_clone, y)

test_1()

import torch
from torch._inductor.bisect_helper import BisectionManager

def test1_fn(x, y):
    x = x + 10
    y[x] += y[x]

def test_1():

    def test():
        torch._dynamo.reset()
        x = torch.randint(-10, -9, (1 ,2),  dtype=torch.int64)
        y = torch.randn((2, 32), dtype=torch.float32)
        x_clone = x.clone()
        y_clone = y.clone()
        with torch.no_grad():
            torch.compile(test1_fn)(x_clone, y_clone)
            test1_fn(x, y)

            return torch.allclose(y_clone, y)

    print(BisectionManager.do_bisect(test))

test_1()

a = torch.randn(2, 32)
b = torch.tensor([3, 4]).to(torch.float)
x = torch.randint(-10, -9, (2,),  dtype=torch.int64) + 10
a.index_put_((x, x), b, accumulate=False)