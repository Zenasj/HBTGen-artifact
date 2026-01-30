import torch
import torch.nn as nn
import numpy as np

class min_model():
    def __init__(self):

        self.class_p        = nn.Parameter(torch.Tensor(np.ones(81) * np.log(1.0)), requires_grad=True)
        self.class_p_t      = self.class_p.data
 
    def prepare_data(self):
        pass

mp.spawn(demo_fn,
             args=(copy.deepcopy(model), world_size,),
             nprocs=world_size,
             join=True)

def demo_basic(rank, queue, world_size):
    model = copy.deepcopy(queue.get())
    print(f'entering demo_basic {rank}')
    setup(rank, world_size)


    # lets use 2 gpus not all of them
    n = 2 // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    model.to(rank)
    ddp_model  = DDP(model, device_ids=[rank], find_unused_parameters=True)
    loss_fn    = nn.MSELoss()
    optimizer  = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs    = ddp_model(torch.randn(20, 10))
    labels     = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()
    print(f'leaving demo_basic {rank}')


def run_demo(demo_fn, world_size):
    model = ToyModel()

    context = mp.get_context("spawn")
    queue = context.Queue()
    queue.put(model)
    queue.put(model)
    mp.spawn(demo_fn,
             args=(queue, world_size,),
             nprocs=world_size,
             join=True)