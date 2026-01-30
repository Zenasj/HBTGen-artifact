import torch.nn as nn

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed.device_mesh import DeviceMesh
from torch import nn

def run(rank, world_size):
    # Set environment variables for distributed setup
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize the process group (using NCCL for GPUs)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Create a device mesh across all GPUs in the process group.
    # This mesh will be identical on every process.
    device_ids = list(range(world_size))  # e.g., [0, 1] for a 2-GPU setup
    mesh = DeviceMesh("cuda", device_ids)
    
    # Define a simple MLP model.
    class ToyModel(nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.linear1 = nn.Linear(8, 8)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(8, 8)

        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    model = ToyModel().cuda()

    # Define a parallelization plan:
    # - Partition the weights of "linear1" column-wise (i.e. split columns across devices).
    # - Partition the weights of "linear2" row-wise (i.e. split rows across devices).
    parallelize_plan = {
        "linear1": ColwiseParallel(),
        "linear2": RowwiseParallel(),
    }

    # Apply tensor parallelism according to the plan.
    tp_model = parallelize_module(model, parallelize_plan=parallelize_plan, device_mesh=mesh)

    # Print the parameter shapes and sample values to verify sharding.
    print(f"Rank {rank} weight distributions:")
    for name, param in tp_model.named_parameters():
        print(f"  {name}: shape = {param.shape}, device = {param.device}")
        # Print a snippet of the flattened parameter values.
        print(f"    sample values: {param.view(-1)[:4]}")

    # Compile the model using torch.compile with the "inductor" backend.
    compiled_model = torch.compile(tp_model, backend="inductor")

    # Dummy input for a forward pass.
    input_tensor = torch.ones(4, 8).cuda()

    # Run the forward pass.
    output = compiled_model(input_tensor)
    print(f"Rank {rank} output: {output}")
    
    # Clean up the process group.
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2  # Adjust based on the number of GPUs you have.
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)