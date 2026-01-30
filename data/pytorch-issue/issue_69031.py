import contextlib
import os

import torch
import torch.distributed
import torch.multiprocessing
import torch.nn as nn

GRADIENT_ACCUMULATION_STEPS = 2


class SimpleConditionalModel(nn.Module):
    # uses nn1 layer on the first pass; nn2 layer on the second pass

    def __init__(self):
        super().__init__()

        self.nn1 = nn.Linear(1, 1)
        self.nn2 = nn.Linear(1, 1)

        self.loss = nn.MSELoss()

        self.state = 0

    def forward(self, input):
        if self.state == 0:
            self.state = 1
            return self.nn1(input)
        else:
            self.state = 0
            return self.nn2(input)


def test_model(rank):
    os.environ["RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29000"

    torch.distributed.init_process_group(backend='gloo', world_size=2)
    model = SimpleConditionalModel()
    ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    for microbatch_idx in range(GRADIENT_ACCUMULATION_STEPS):
        context = contextlib.nullcontext
        if microbatch_idx < GRADIENT_ACCUMULATION_STEPS - 1:
            context = ddp.no_sync

        with context():
            input = torch.rand((1, ))
            output = ddp.forward(input)
            target = torch.rand((1, ))

            loss = model.loss(output, target)
            loss.backward()

    print(f'Rank {rank} gradients: {[p.grad for p in model.parameters()]}')


if __name__ == '__main__':
    torch.multiprocessing.spawn(test_model, nprocs=2)