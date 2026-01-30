import torch.nn as nn

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
import torch.distributed as dist
import os

rank, world_size = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])

print((f'Torch version is {torch.__version__}' if rank==0 else ''))

torch.cuda.set_device(f'cuda:{rank}')
dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 4, bias=False),
        )

unwrapped_model = MyModel().cuda()

wrapped_model = FSDP(
    module=unwrapped_model,
    use_orig_params=True
)

with FSDP.state_dict_type(module=wrapped_model, state_dict_type=StateDictType.LOCAL_STATE_DICT):
    sd = wrapped_model.state_dict()
    print(f'\nState Dict for Rank {rank}:')
    for k,v in sd.items():
        print('\t' + k + ': ', v.shape)
        
    # This line errors out because there are extra parameters in the state_dict.
    wrapped_model.load_state_dict(sd, strict=True)