python
import torch
import torch.utils.checkpoint as cp

param = torch.tensor(1., requires_grad=True)

def f(x):
    print("f called")
    return (x + param)[:].relu()

def g(x):
    print("g called")
    return (x + param).relu_()

def h(x):
    print("h called")
    return (x + param)[:].relu_()  # problematic


for func in [f, g, h]:
    param.grad = None
    x = torch.zeros(5, requires_grad=True)
    y = cp.checkpoint(func, x)
    print(f"grad_fn: {y.grad_fn}")
    y.sum().backward()
    print(f"param.grad: {param.grad}\n")

python
def robust_checkpoint(function, *args, **kwargs):
    def wrapper(*args_, **kwargs_):
        result = function(*args_, **kwargs_)
        if isinstance(result, torch.Tensor):
            return result[...] if result._version > 0 else result
        else:
            raise NotImplementedError()

    return cp.checkpoint(wrapper, *args, **kwargs)