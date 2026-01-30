import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    embed_dim = 128
    kv_embed_dim = 1024
    num_heads = 8
    model = torch.nn.MultiheadAttention(embed_dim, num_heads,
                                        kdim=kv_embed_dim,
                                        vdim=kv_embed_dim).to(device_ids[0])

    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    # inputs
    src_len, tgt_len, batch_size = 10, 5, 16
    query = torch.rand((tgt_len, batch_size, embed_dim)).to(device_ids[0])
    key = torch.rand((src_len, batch_size, kv_embed_dim)).to(device_ids[0])
    value = key.to(device_ids[0])

    outputs = ddp_model(query, key, value)

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 2)

def demo_basic(rank, world_size):
    setup(rank, world_size)

    # setup devices for this process, rank 1 uses GPUs [0, 1, 2, 3] and
    # rank 2 uses GPUs [4, 5, 6, 7].
    n = torch.cuda.device_count() // world_size
    device_ids = list(range(rank * n, (rank + 1) * n))

    # create model and move it to device_ids[0]
    embed_dim = 128
    kv_embed_dim = 1024
    num_heads = 8
    model = torch.nn.MultiheadAttention(embed_dim, num_heads,
                                        kdim=kv_embed_dim,
                                        vdim=kv_embed_dim)

    # output_device defaults to device_ids[0]
    ddp_model = DDP(model, device_ids=device_ids)

    # inputs
    src_len, tgt_len, batch_size = 10, 5, 16
    query = torch.rand((tgt_len, batch_size, embed_dim))
    key = torch.rand((src_len, batch_size, kv_embed_dim))
    value = key

    outputs = ddp_model(query, key, value)[0]
    loss_fn = torch.nn.functional.l1_loss

    loss = loss_fn(outputs, torch.zeros_like(outputs))

    loss.backward()
    ddp_model.zero_grad()

    outputs = ddp_model(query, key, value)[0]

    cleanup()