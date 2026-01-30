import torch
import torch.nn as nn
import torch.distributed.fsdp as fsdp
import torch.distributed as dist

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class MyBlock(nn.Module):
    def __init__(self):
        super(MyBlock, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)

def test_fsdp_ignored_module_meta(rank):
        class CPUGPUModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(1, 1)
                self.b = MyBlock()
        with torch.device("meta"):
            m= CPUGPUModule()
        m = FSDP(m, device_id=rank, ignored_modules=[m.a, m.b.fc1], use_orig_params=True, auto_wrap_policy=fsdp.wrap.ModuleWrapPolicy({MyBlock}))
        print(f"RV: {next(m.a.parameters()).device}")
        print(f"RV: {next(m.b.parameters()).device}")
        '''
        RV: meta
        RV: cuda:0
        '''

dist.init_process_group(backend='nccl')

test_fsdp_ignored_module_meta(dist.get_rank())

dist.destroy_process_group()