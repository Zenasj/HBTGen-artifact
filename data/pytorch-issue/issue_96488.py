import torch
import contextlib
import torch._dynamo

import logging
torch._dynamo.config.log_level = logging.DEBUG
torch._dynamo.config.output_code = True

@contextlib.contextmanager
def ctx():
    try:
        yield
    except RuntimeError:
        print("out")

@torch._dynamo.optimize("eager")
def f():
    with ctx():
        h()

def h():
    raise RuntimeError("boof")

f()