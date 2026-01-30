import torch

init_rpc("worker0", rank=0)
# send to worker1 which joins after worker0
rpc.sync("worker1", torch.add, (torch.tensor(1), torch.tensor(1)))

# joins after worker0
init_rpc("worker1", rank=1)