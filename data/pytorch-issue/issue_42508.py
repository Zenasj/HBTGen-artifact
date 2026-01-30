import torch.nn as nn

def __init__(
        self, module: nn.Module, oss: OSS, world_size: int, process_group: Any = None, buffer_size: int = 2 ** 28
    ):
        super().__init__()

        self.module = module
        self.world_size = world_size
        self.process_group = process_group if process_group is not None else dist.group.WORLD
        self.rank = dist.get_rank(self.process_group)