import torch
from torch._dynamo.comptime import comptime
import contextlib
import logging
torch._dynamo.config.log_level = logging.DEBUG

@torch._dynamo.optimize("eager")
def h():
    x = set()
    comptime.graph_break()
    x.add(1)
    return x

h()

set