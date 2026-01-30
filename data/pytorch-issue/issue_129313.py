import torch.nn as nn

import torch
import os
os.environ['NCCL_DEBUG'] = 'WARN'
from torch import nn
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel
from torch.distributed._tensor import Replicate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
backend = 'nccl' if device == 'cuda' else 'gloo'

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
dist.init_process_group(backend=backend, world_size=world_size)
if device == 'cuda':
    torch.cuda.set_device(local_rank)

device_mesh = init_device_mesh(device, [2, 2], mesh_dim_names=['dp', 'tp'])
tp_mesh = device_mesh['tp']

def print_on_all_rank(string=None, title=None):
    rank = dist.get_rank()
    if rank == 0:
        if title is not None:
            print(title)
    dist.barrier()
    for i in range(world_size):
        if i == rank:
            print(f'Global rank: {rank}, tp_rank: {device_mesh["tp"].get_local_rank()}, dp_rank: {device_mesh["dp"].get_local_rank()}')
            if string is not None:
                print(string)
        dist.barrier()
    if rank == 0:
        print()
    dist.barrier()

with torch.device(device):
    model = nn.Embedding(num_embeddings=4, embedding_dim=1)
    # hardcode the weights to visualize and reproduce
    model.weight.data[0, 0] = 0.
    model.weight.data[1, 0] = 1.
    model.weight.data[2, 0] = 2.
    model.weight.data[3, 0] = 3.


# apply tp
model = parallelize_module(
    model,
    tp_mesh,
    parallelize_plan=RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=Replicate(),  # reshard on sequence parallel in tp group
    ),
)

# apply fsdp2
from torch.distributed._composable.fsdp import fully_shard

model = fully_shard(model, mesh=device_mesh['dp'], reshard_after_forward=True)

fsdp_state = fully_shard.state(model)
fsdp_state._lazy_init()

model.unshard()

# print embedding weights on each rank
print_on_all_rank(model.weight, '======Weights after unshard======')

from torch.distributed.checkpoint.state_dict import get_model_state_dict
state_dict = get_model_state_dict(model)
print_on_all_rank(state_dict, '======State dict after calling get_model_state_dict=======')

# print embedding weights on each rank
print_on_all_rank(model.weight, '======Weights after unshard after calling get_model_state_dict======')

full_tensor = model.weight.full_tensor()

if rank == 0:
    print('======Full tensor=======')
    print(full_tensor)

dist.barrier()