import torch

from torch.utils.collect_env import get_running_cuda_version, run
print(get_running_cuda_version(run))

None