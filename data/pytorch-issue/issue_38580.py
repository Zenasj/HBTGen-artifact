# Helper functions
def get_local_rank():
    return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])

def init_rpc():
    world_size=2
    rank=get_local_rank()
    os.environ['MASTER_ADDR'] = '10.123.134.28'
    os.environ['MASTER_PORT'] = '21234'
    if rank == 1:
        rpc.init_rpc("rank1", rank=rank, world_size=world_size)
    else:
        rpc.init_rpc("rank0", rank=rank, world_size=world_size)

def shutdown_rpc():
    rpc.shutdown()

# Define RPC-based model with 2 modules being remote used by DistModelParallel main model
class Net1(nn.Module):
    def __init__(self):
        pass
    def forward(self, input):
        return input

class Net2(nn.Module):
    def __init__(self,):
        pass
    def forward(self, input):
        return input

class DistModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = rpc.remote("rank0", Net1)
        self.net2 = rpc.remote("rank1", Net2)
       
    def forward(self, input):
        outputs1_rref = rpc.rpc_async(self.net1, Net1.forward, args=input)
        # outputs2_rref = rpc.rpc_async(self.net2, Net2.forward, args=(outputs1_rref))
        # return outputs2_rref
        return input

# Repro'ing the issue
init_rpc()
rank = get_local_rank()
if rank == 1:
    dmp = DistModelParallel()
    scripted_model = torch.jit.script(dmp)
    print(f'{"*"*80}\n{scripted_model.code}\n{"-"*80}')
shutdown_rpc()

from torch import Tensor
import torch.nn as nn

class Net1(nn.Module):
    def __init__(self):
        pass

    def forward(self : nn.Module, input):
        return input

class Net1(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        return input

class Net2(nn.Module):
    def __init__(self,):
        pass

    def forward(self, input):
        return input

class DistModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = rpc.remote("rank0", Net1)
        self.net2 = rpc.remote("rank1", Net2)
        
    def forward(self, input):
        outputs1_rref = rpc.rpc_async(self.net1, Net1.forward, (input, input))
        outputs1 = outputs1_rref.wait()
        outputs2_rref = rpc.rpc_async(self.net2, Net2.forward, (outputs1, outputs1))
        outputs2 = outputs2_rref.wait()
        return outputs2