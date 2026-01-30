import inspect
import torch


def greet(greeting, name, punctuation='!'):
    """Simple function to greet a person."""
    print(f"{greeting}, {name}{punctuation}")

# Obtain the signature of the function
sig = inspect.signature(greet)


def fn(x):
    sig.bind("Hello", "Alice")
    return torch.sin(x)

opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
x = torch.randn(3)
opt_fn(x)