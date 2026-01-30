import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")
import pdb
import torch
from torch import nn
import torch.distributed as dist
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)
dist.init_process_group(backend="nccl")
dist.barrier()
global_rank = dist.get_rank()
torch.cuda.set_device(global_rank)
print(global_rank)
#
#
#
# setting up FSDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
mixed_precision_config = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
sharding_strategy_config = ShardingStrategy.SHARD_GRAD_OP
model1 = nn.Linear(24, 24)
model2 = nn.Linear(24, 24)
sharded_model = FSDP(
    model1, 
    sharding_strategy=sharding_strategy_config,
    mixed_precision=mixed_precision_config,
    device_id=global_rank, 
    use_orig_params=True,
    # auto_wrap_policy=ModuleWrapPolicy({Block}),
)
x_half = torch.randn(32,24).cuda().half()
x_half.requires_grad_()
sharded_model(x_half).sum().backward()

with FSDP.state_dict_type(sharded_model, StateDictType.LOCAL_STATE_DICT):
    sd = sharded_model.state_dict()
    if global_rank==0:
        print(sd.keys())
    torch.save(sd, f"/tmp/tmp_rank{global_rank}.pth")

torch.distributed.barrier()
t = torch.load(f"/tmp/tmp_rank{global_rank}.pth", map_location="cpu")