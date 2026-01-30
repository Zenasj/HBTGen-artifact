import torch

def f():
    getattr(None, 'arg', 3)

import logging

from torch._dynamo import config, explain

config.verbose = True
config.log_level = logging.DEBUG

config.repro_after = "dynamo"
config.repro_level = 3

config.output_code = True
config.output_graph_code = True
config.print_graph_breaks = True


def f():
    getattr(None, 'arg', 3)

##### fails
explain(f)

##### also fails
torch.compile()(f)()