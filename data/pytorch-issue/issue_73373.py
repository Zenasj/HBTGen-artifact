import torch

init_rpc("worker0", rank=0)

init_rpc("worker1", rank=1)
rpc.sync("worker0", torch.add, (torch.tensor(1), torch.tensor(1)))