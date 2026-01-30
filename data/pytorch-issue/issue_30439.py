import torch.nn as nn

import os
import torch

class Toy(torch.nn.Module):
    """Toy model"""
    def __init__(self):
        """__init__"""
        super(Toy, self).__init__()
        self.L = torch.nn.Linear(1, 50)
        self.out = torch.nn.Linear(50, 1)

    def forward(self, x):
        """forward"""

        Y = torch.tanh(self.L(x))
        Y = self.out(Y)
        return Y

if __name__ == "__main__":

    # some misc settings
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    # communication setup
    torch.distributed.init_process_group("gloo")
    my_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # setup data loaders
    delta = 1. / world_size
    start, end = delta * my_rank, delta * (my_rank + 1)
    x = torch.linspace(start, end, 100//world_size)[:, None]
    y = 2.31 * torch.pow(x, 2) + 1.23 * x - 3.21

    # model, loss, optimizer
    model = torch.nn.parallel.DistributedDataParallel(Toy())
    loss_fn = torch.nn.MSELoss(reduction="mean")
    opter = torch.optim.LBFGS(model.parameters(), line_search_fn="strong_wolfe")

    for epoch in range(3):

        def closure():
            model.train()
            opter.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            return loss

        opter.step(closure)
        print("rank {}: Epoch {} done".format(my_rank, epoch))

    print("rank {}: training finished".format(my_rank))

opter = torch.optim.LBFGS(
    model.parameters(), tolerance_grad=0, tolerance_change=0, line_search_fn=None)