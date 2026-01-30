import torch

from torch._dynamo import compiled_autograd
def compiler_fn(gm):
        return torch.compile(gm, fullgraph=True, backend="inductor")
        with compiled_autograd.enable(compiler_fn):
                train_step()