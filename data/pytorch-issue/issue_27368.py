# On caller
import torch
import torch.distributed as dist

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=1)
dist.init_model_parallel("worker0")

def my_tensor_function(a, b):
    return a + b

ret = dist.rpc_sync("worker1", my_tensor_function, args=(torch.ones(2, 2), torch.ones(2, 2)))
dist.join_rpc()

# On callee
import torch.distributed as dist

#def my_tensor_function(a, b):
#        return a + b

dist.init_process_group("gloo", init_method='tcp://127.0.0.1:12356', world_size=2, rank=0)
dist.init_model_parallel("worker1")
dist.join_rpc()