import torch

from torch._inductor import config
config.triton.mm = "autotune"