def reporter(*args, **kwargs):
    print("in reporter")

import torch
torch._C._cuda_attach_out_of_memory_observer(reporter)