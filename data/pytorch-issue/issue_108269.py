import torch.nn as nn

#!/usr/bin/env python
from functools import partial
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)


class MyModel(torch.nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()
      self.fc1 = torch.nn.Linear(64, 32)
      self.fc2 = torch.nn.Linear(32, 16)
      self.fc3 = torch.nn.Linear(16, 8)
    
    def forward(self, inp):
      return self.fc3(self.fc2(self.fc1(inp)))

def run(rank, size):
    """ Distributed function to be implemented later. """
    model = MyModel().to(rank)

    # Activation checkpointing for Linear layers.
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, torch.nn.Linear)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    ddp = torch.compile(DDP(model, device_ids=[rank]))
    out = ddp(torch.rand(10, 64).cuda(rank))
    out.sum().backward()
    print(f"{rank=} done")

def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

mod._param_name_to_source={'self.fc1._checkpoint_wrapped_module.weight': ParamBufferSource(base=AttrSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc1'), member='_checkpoint_wrapped_module'), member='weight'), 'self.fc1._checkpoint_wrapped_module.bias': ParamBufferSource(base=AttrSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc1'), member='_checkpoint_wrapped_module'), member='bias'), 'self.fc2._checkpoint_wrapped_module.weight': ParamBufferSource(base=AttrSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc2'), member='_checkpoint_wrapped_module'), member='weight'), 'self.fc2._checkpoint_wrapped_module.bias': ParamBufferSource(base=AttrSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc2'), member='_checkpoint_wrapped_module'), member='bias'), 'self.fc3._checkpoint_wrapped_module.weight': ParamBufferSource(base=AttrSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc3'), member='_checkpoint_wrapped_module'), member='weight'), 'self.fc3._checkpoint_wrapped_module.bias': ParamBufferSource(base=AttrSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc3'), member='_checkpoint_wrapped_module'), member='bias'), 'L__self___fc3.weight': ParamBufferSource(base=NNModuleSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc3')), member='weight'), 'L__self___fc3.bias': ParamBufferSource(base=NNModuleSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc3')), member='bias'), 'L__self___fc2.weight': ParamBufferSource(base=NNModuleSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc2')), member='weight'), 'L__self___fc2.bias': ParamBufferSource(base=NNModuleSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc2')), member='bias'), 'L__self___fc1.weight': ParamBufferSource(base=NNModuleSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc1')), member='weight'), 'L__self___fc1.bias': ParamBufferSource(base=NNModuleSource(base=AttrSource(base=LocalSource(local_name='self', cell_or_freevar=False), member='fc1')), member='bias')}

def forward(self, inp):
      fc1 = torch.utils.checkpoint.checkpoint(self.fc1, inp, use_reentrant=False)
      fc2 = torch.utils.checkpoint.checkpoint(self.fc2, fc1, use_reentrant=False)
      fc3 = torch.utils.checkpoint.checkpoint(self.fc3, fc2, use_reentrant=False)
      return fc3