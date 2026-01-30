import torch

3
import functools
from timeit import default_timer as timer

from torch.utils.data.datapipes.utils.common import validate_input_col


def foo(*args):
    pass

d = {i: list(range(i)) for i in range(10_000)}
partial_foo = functools.partial(foo, d)

start = timer()
validate_input_col(fn=partial_foo, input_col=[1, 2])
end = timer()
print(f"elapsed time: {round(end - start, 2)}")