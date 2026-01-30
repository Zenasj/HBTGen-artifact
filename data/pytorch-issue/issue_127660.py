import torch

torch._inductor.config.debug = True
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_dir = "..."