import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from linformer_pytorch import LinformerLM
from torch.nn.parallel import DistributedDataParallel as DDP


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    
    model = LinformerLM(
            num_tokens=30522,  # Number of tokens in the LM
            input_size=5120,  # Dimension 1 of the input
            channels=128,  # Dimension 2 of the input
            dim_d=None,
            # Overwrites the inner dim of the attention heads. If None, sticks with the recommended channels // nhead, as in the "Attention is all you need" paper
            dim_k=128,  # The second dimension of the P_bar matrix from the paper
            dim_ff=128,  # Dimension in the feed forward network
            dropout_ff=0.15,  # Dropout for feed forward network
            nhead=16,  # Number of attention heads
            depth=12,  # How many times to run the model
            dropout=0.1,  # How much dropout to apply to P_bar after softmax
            activation="gelu",
            # What activation to use. Currently, only gelu and relu supported, and only on ff network.
            checkpoint_level="C2",  # What checkpoint level to use. For more information, see below.
            parameter_sharing="none",  # What level of parameter sharing to use. For more information, see below.
            k_reduce_by_layer=0,
            # Going down `depth`, how much to reduce `dim_k` by, for the `E` and `F` matrices. Will have a minimum value of 1.
            full_attention=False,
            # Use full attention instead, for O(n^2) time and space complexity. Included here just for comparison
            include_ff=True,  # Whether or not to include the Feed Forward layer
            w_o_intermediate_dim=None,
            # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
            emb_dim=128,  # If you want the embedding dimension to be different than the channels for the Linformer
        ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randint(20000, (3, 5120)))
    labels = torch.randint(20000, (3, 5120)).to(rank)
    loss_mx = labels != -100
    output = outputs[loss_mx].view(-1, 30522)
    labels = labels[loss_mx].view(-1)
    loss_fn(output, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
if __name__ == "__main__":
    run_demo(demo_basic, 2)