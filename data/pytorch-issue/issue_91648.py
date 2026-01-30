import torch.nn as nn

from typing import Iterable, List

from torch import nn


class MyDataParallel(nn.DataParallel):
    def replicate(self, module: nn.Module, device_ids: Iterable[int]) -> List[nn.Module]:
        replicas = super().replicate(module, device_ids)
        ...
        return replicas