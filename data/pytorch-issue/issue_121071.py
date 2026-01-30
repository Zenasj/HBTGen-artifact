import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 4, 3, 1)
        self.fc1 = nn.Linear(576, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(input, target, model, rank, world_size, optimizer):
    output = model(input)
    loss = F.nll_loss(output, target, reduction='sum')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fsdp_main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device("cuda:{}".format(rank))
    model = Net().to("cuda:{}".format(rank))
    model = FSDP(model, device_id="cuda:{}".format(rank), use_orig_params=True).train()
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    model = torch.compile(model)

    input = torch.randn(4, 1, 28, 28).to("cuda:{}".format(rank))
    target = torch.randint(1, 10, (4,)).to("cuda:{}".format(rank))

    with torch._dynamo.compiled_autograd.enable(torch.compile):
        train(input, target, model, rank, world_size, optimizer)
    dist.destroy_process_group()

if __name__ == '__main__':
    WORLD_SIZE = 2
    mp.spawn(fsdp_main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)

torch._inductor.config.reorder_for_compute_comm_overlap = True
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ["sink_waits", "raise_comms"]
torch._inductor.config.allow_buffer_reuse = True
torch._inductor.config.compile_threads = 1

@torch.compile()
def train(a):
    c = torch.relu(a)
    d = torch.matmul(c, c)
    ar = _functional_collectives.all_reduce(a, "sum", [0, 1], "")
    e = d + ar
    return e

torch._inductor.config.reorder_for_compute_comm_overlap = True
torch._inductor.config.reorder_for_compute_comm_overlap_passes = ["raise_comms"]
torch._inductor.config.allow_buffer_reuse = True
torch._inductor.config.compile_threads = 1

@torch.compile()
def train(a):
    c = torch.matmul(a, a)
    g = torch.matmul(c, c)
    ar = _functional_collectives.all_reduce(a, "sum", [0, 1], "")
    x = torch.matmul(g, g)
    z = x + ar
    return z