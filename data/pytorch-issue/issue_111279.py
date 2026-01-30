import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

torch.set_float32_matmul_precision('high')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ToyModel(nn.Module):

    def __init__(self,):
        super().__init__()
        layers = []
        for l in range(2):
            layer = nn.ModuleList([nn.LayerNorm(96), nn.MultiheadAttention(embed_dim=96, num_heads=4, batch_first=True)])
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H = x.shape

        for m in self.layers:
            x = x.reshape(B * F, T, H)
            x = m[0](x)
            x, attn = m[1].forward(x, x, x)
            x = x.reshape(B, F, T, H)

        return x


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    model.compile()  # if comment this line, the training process will be OK
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(2, 129, 100, 96))
    labels = torch.randn(2, 129, 100, 96).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()

    print("success")


if __name__ == '__main__':
    world_size = 2
    mp.spawn(demo_basic, args=(world_size,), nprocs=world_size, join=True)