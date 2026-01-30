import itertools
import torch

def fn() -> None:
    a = list(range(1, 3))
    b = list(range(3, 6))
    c = [a, b]
    d = itertools.chain.from_iterable(c)
    return d


comp = torch._dynamo.optimize(nopython=True)(fn)()