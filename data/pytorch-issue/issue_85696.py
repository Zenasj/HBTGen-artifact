import torch

class ErrorMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(
        self,
        orig_func,
        types,
        args = (),
        kwargs = None,
    ):
        if kwargs is None:
            kwargs = {}
        raise RuntimeError("test")
        return orig_func(*args, **kwargs)

class PassThrough(torch.overrides.TorchFunctionMode):
    def __torch_function__(
        self,
        orig_func,
        types,
        args = (),
        kwargs = None,
    ):
        if kwargs is None:
            kwargs = {}
        return orig_func(*args, **kwargs)

# Taken from
# https://pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html
class LegendrePolynomial3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        with ErrorMode():
            result = grad_output * 1.5 * (5 * input ** 2 - 1)
        return result

P3 = LegendrePolynomial3.apply
x = torch.randn(2, requires_grad=True)

def func(x):
    y = P3(x)
    return torch.autograd.grad([y], [x], [torch.ones_like(y)])

try:
    func(x)
except RuntimeError as e:
    print(f"Error message: {e}")

# Should raise "RuntimeError: test" but doesn't
try:
    with PassThrough():
        func(x)
        print("No error!")
except RuntimeError as e:
    print(f"Error message: {e}")