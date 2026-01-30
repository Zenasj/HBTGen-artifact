import torch
import torch.distributed.rpc as rpc
rpc.init_rpc("worker0", rank=0, world_size=1)