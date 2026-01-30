import os
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo"

import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.distributed.tensor.parallel import *
from torch.distributed._tensor import DTensor, Shard, Replicate, distribute_tensor, distribute_module, init_device_mesh

dist.init_process_group()
global_rank = dist.get_rank()
device = global_rank % torch.cuda.device_count()
torch.set_default_device(device)
world_size = dist.get_world_size()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(50, 80)
        self.fc2 = nn.Linear(80, 50)
        self.fc3 = nn.Linear(50, 32)
        self.fc4 = nn.Linear(32, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        h1 = nn.ReLU()(self.fc1(x))
        h2 = nn.ReLU()(self.fc2(h1))
        h3 = nn.ReLU()(self.fc3(h2))
        out = self.fc4(h3)
        if target is not None:
            loss = self.loss_fn(out, target)
        return (out, loss)

device_mesh = init_device_mesh("cuda", (world_size,))
plan = {
    "fc1": ColwiseParallel(output_layouts=Shard(1)),
    "fc2": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(1)),
    "fc3": RowwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
    "fc4": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=False)
}

net = Net()
net = parallelize_module(net, device_mesh, plan)
net = torch.compile(net)

net.train()
for i in range(5):
    print(f"[rank{global_rank}] Running iteration {i}")
    x = torch.rand(4, 50)
    target = torch.empty(4, dtype=torch.long).random_(10)
    x = x.to(global_rank)
    target = target.to(global_rank)
    x = distribute_tensor(x, device_mesh=device_mesh, placements=[Replicate()])
    target = distribute_tensor(target, device_mesh=device_mesh, placements=[Replicate()])
    with loss_parallel():
        (_, loss) = net(x, target=target)
        loss.backward()


print("[rank{global_rank}] ]Completed!")