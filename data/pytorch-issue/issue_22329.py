import torch

@torch.jit.autograd_script
def my_function(x):
    def backward(grad_out):
        return 99 * x * grad_out  # not the true gradient, mind you!
    return x**2, backward

@torch.jit.script
def test(x):
    res = my_function(x)
    return res