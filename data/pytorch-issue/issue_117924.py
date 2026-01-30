import torch
import torch.nn as nn
import torch.distributed.fsdp as fsdp
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def test_fsdp_ignored_module_meta(rank):
        class CPUGPUModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(1, 1)
        with torch.device("meta"):
            m= CPUGPUModule()
        m = FSDP(m, device_id=rank, ignored_states=[m.a.weight], use_orig_params=True)
        print(f"RV: {next(m.a.parameters()).device}")
        '''
        RV: cuda:0
        '''

dist.init_process_group(backend='nccl')

test_fsdp_ignored_module_meta(dist.get_rank())

dist.destroy_process_group()

RV: meta