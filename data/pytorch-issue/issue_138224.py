from omegaconf import DictConfig
import torch

@torch.compile(fullgraph=True)
def func(cfg):
    return cfg.param + 1

cfg = DictConfig({"param": 0})

func(cfg) # breaks because `__getattr__` is not supported