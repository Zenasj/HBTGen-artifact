import torch

class MyClass:
    i = 3

    @staticmethod
    def foo_inner(x):
        return torch.mul(x, MyClass.i)

    # allow_in_graph silently doesn't work, dynamo inlines foo1
    @staticmethod
    @torch._dynamo.allow_in_graph
    def foo1(x):
        return MyClass.foo_inner(x)

# allow_in_graph seems to work here though
@torch._dynamo.allow_in_graph
def foo2(x):
    return MyClass.foo_inner(x)





@torch.compile(backend="eager")
def f_bad(x):
    return MyClass.foo1(x)

@torch.compile(backend="eager")
def f_good(x):
    return foo2(x)

x = torch.ones(2)
f_bad(x)
f_good(x)