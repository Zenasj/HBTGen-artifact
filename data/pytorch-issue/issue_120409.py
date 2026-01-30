import torch

def foo(x):
    assert x.shape[0]>4
    return x+1

print("Raises no exception:",torch.compile(foo)(torch.zeros(2))) # doesn't fail!  this is bad!
print("This will raise an exception:",foo(torch.zeros(2))) # fails!  this is good!  without compiilation, error is raised as expected

py
import torch

def foo(x):
    assert x.shape[0]>4
    return x+1

torch.compile(foo, backend="aot_eager")(torch.zeros(2))  # raises error