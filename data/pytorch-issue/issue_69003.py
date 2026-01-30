import torch.nn as nn

import torch.nn.parallel
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = str(os.environ.get('HOST', '127.0.0.1'))
os.environ['MASTER_PORT'] = str(os.environ.get('PORT', 29500))
os.environ['RANK'] = str(os.environ.get('SLURM_LOCALID', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('SLURM_NTASKS', 2))

backend = os.environ.get('BACKEND', 'mpi')
print('Using backend:', backend)

dist.init_process_group(backend)
# dist.init_process_group(backend, init_method=f"tcp://{master_add}:{master", rank=rank, world_size=size)
my_rank = dist.get_rank()
my_size = dist.get_world_size()

print("my rank = %d  my size = %d" % (my_rank, my_size))

dist.destroy_process_group()