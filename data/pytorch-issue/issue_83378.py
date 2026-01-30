import torch

def check_no_grad(f):
    f_ = torch.jit.script(f)

    y = torch.ones(1)
    t = torch.ones(1)
    assert not t.requires_grad

    y.requires_grad_()
    assert y.requires_grad
    assert f_(y, t).requires_grad

    y.requires_grad_(False)
    assert not y.requires_grad
    if f_(y, t).requires_grad:
        print(f.__name__, "fails")
    else:
        print(f.__name__, "works")

def works_1(a, b):
    x = b / (-b)
    return a + x

def works_2(a, b):
    x = (b / b)[:]
    return a + x

def works_3(a, b):
    x = -(b / b)[:]
    return a + x

def fail_1(a, b):
    x = (b / (-b))[:]
    return a + x

check_no_grad(works_1)
check_no_grad(works_2)
check_no_grad(works_3)
check_no_grad(fail_1)