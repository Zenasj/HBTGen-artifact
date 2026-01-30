import torch
import torchdynamo
from torch import nn
from torch.distributed.fsdp.flat_param import HandleConfig, HandleShardingStrategy
from torch.distributed.fsdp.flatten_params_wrapper import FlattenParamsWrapper
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 10000),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class Wrapper(nn.Module):
    def __init__(self, wrapped):
        super(Wrapper, self).__init__()
        self.mod = wrapped
    
    def forward(self, *args, **kwargs):
        return self.mod(*args, **kwargs)

def flatten():
    mod = ToyModel()
    # FlattenParamsWrapper is a helper class used inside FSDP.  It appears to not work with Dynamo
    mod = FlattenParamsWrapper(
        mod,
        params=list(mod.parameters()),
        device="cpu",
        config=HandleConfig(HandleShardingStrategy.FULL_SHARD, False, None, None),
    )
    mod = torchdynamo.optimize("aot_eager")(mod)
    inputs = torch.randn(20, 10)
    outputs = mod(inputs)
    print(outputs)



if __name__ == "__main__":
    flatten()
    """
    Currently getting this error:

    Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. Please convert all Tensors to FakeTensors first. Found in aten.t.default(*(tensor([[ 0.2899, -0.1847,  0.0950,  ...,  0.0304, -0.2226, -0.3076],
        [-0.1424,  0.2049, -0.2806,  ..., -0.0127, -0.0831,  0.0546],
        [ 0.1324, -0.0840, -0.0635,  ..., -0.0338, -0.1047,  0.2138],
        ...,
        [ 0.2627,  0.1499,  0.0301,  ..., -0.2043, -0.0677,  0.0513],
        [ 0.0060, -0.1727, -0.1449,  ..., -0.0631, -0.0106, -0.0005],
        [-0.0464, -0.1969,  0.1693,  ...,  0.1114,  0.0926, -0.2426]],
       grad_fn=<ViewBackward0>),), **{})

    
    """

flat_tensor = torch.zeros(100)
orig_param1.data = flat_tensor[:10]
orig_param2.data = flat_tensor[10:20]
...
orig_param10.data = flat_tensor[90:]

import torch
import torchdynamo
import torch.nn as nn



class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10000)

    def forward(self, x):
        return self.linear(x * 4)



mod = ToyModel()
print(getattr(mod, "linear").weight.shape)
delattr(getattr(mod, "linear"), "weight")
# Set linear.weight to be a fresh **Parameter**, not a tensor!
setattr(getattr(mod, "linear"), "weight", torch.nn.Parameter(torch.randn(10000, 10)))


opt_mod = torchdynamo.optimize("aot_eager")(mod)

a = torch.randn(20, 10)

ref = mod(a)
# runs without error
res = opt_mod(a)

import torch
import torchdynamo
import torch.nn as nn



class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.linear = nn.Linear(10, 10000)

    def forward(self, x):
        return self.linear(x * 4)



mod = ToyModel()
mod_linear = getattr(mod, "linear")
delattr(mod_linear, "weight")
# Set linear.weight to a *buffer* on the module!
mod_linear.register_buffer("weight", torch.randn(10000, 10))


opt_mod = torchdynamo.optimize("aot_eager")(mod)

a = torch.randn(20, 10)

ref = mod(a)
# runs without error
res = opt_mod(a)