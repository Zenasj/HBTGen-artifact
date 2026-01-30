import torch

if torch._C._rpc_init():
    from .optimizer import DistributedOptimizer

from .post_localSGD_optimizer import PostLocalSGDOptimizer
from .zero_redundancy_optimizer import ZeroRedundancyOptimizer