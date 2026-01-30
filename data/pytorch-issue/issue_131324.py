import torch

@torch.compile(backend="aot_eager")
def f(x, y):
    return x * y

def f_wrapper(x, y):
    torch._dynamo.decorators.mark_unbacked(x, 0)
    torch._dynamo.decorators.mark_unbacked(y, 0)
    torch._dynamo.mark_dynamic(x, 1)
    torch._dynamo.mark_dynamic(y, 1)
    return f(x, y)


x = torch.randn(5, 6)
y = torch.randn(5, 6)
_ = f_wrapper(x, y)

# hopefully don't recompile
x = torch.randn(1, 8)
y = torch.randn(1, 8)
_ = f_wrapper(x, y)

repro.py
import torch

@torch.compile(backend="aot_eager")
def f(x, y):
    return x * y

def f_wrapper(x, y):
    torch._dynamo.decorators.mark_unbacked(x, 0)
    torch._dynamo.decorators.mark_unbacked(y, 0)
    torch._dynamo.mark_dynamic(x, 1)
    torch._dynamo.mark_dynamic(y, 1)
    return f(x, y)


x = torch.randn(5, 6)
y = torch.randn(5, 6)
print(f_wrapper(x, y))

# hopefully don't recompile
x = torch.randn(1, 8)
y = torch.randn(1, 8)
print(f_wrapper(x, y))