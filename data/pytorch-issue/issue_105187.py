import torch

from torch.autograd.profiler import profile, _ExperimentalConfig
with profile(use_kineto=True, use_cuda=True,
   experimental_config=_ExperimentalConfig(enable_cuda_sync_events=True),
) as prof:
   workload()