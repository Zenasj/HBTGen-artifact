import torch

torch.distributed.init_process_group(backend="mpi")
nccl_group = torch.distributed.new_group(backend="nccl")