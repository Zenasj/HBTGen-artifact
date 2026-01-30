import torch.nn as nn

import torch
import torchdynamo
import logging

torchdynamo.disallow_in_graph(torch.ops.profiler._record_function_enter)
torchdynamo.disallow_in_graph(torch.ops.aten.relu)
# torchdynamo.disallow_in_graph(torch.relu)

def gn(x, y):
    torch.ops.profiler._record_function_enter("Starting the additions", "idk what this argument does")
    return x * y

# @torchdynamo.optimize("aot_print")
def fn(x, y):
    z = x + y
    r = torch.ops.aten.relu(z)
    z_1 = gn(z, r) + 4
    r_1 = r + 4
    return torch.nn.functional.gelu(z_1 - r_1)

x, y = [torch.rand((10, 10)) for _ in range(2)]

# torchdynamo.config.verbose = True
# torchdynamo.config.log_level = logging.INFO

with torchdynamo.optimize("aot_print"):
    fn(x, y)

# fn(x, y)