import os
import torch
from torch.distributed import rpc
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
opts = rpc.TensorPipeRpcBackendOptions(devices=[]) # empty list, not None
torch.cuda.is_initialized() # returns False
rpc.init_rpc('worker0', world_size=1, rank=0, rpc_backend_options=opts)
torch.cuda.is_initialized() # returns True