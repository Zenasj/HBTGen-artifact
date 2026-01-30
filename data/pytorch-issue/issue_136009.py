import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

def train(rank, world_size):
    # Initialize the process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank
    )

    # Set the device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Create the model and move it to the appropriate device
    model = nn.Linear(10, 1).to(device)

    # Wrap the model in DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Create GradScaler with different init_scale for each rank
    if rank == 0:
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
    else:
        scaler = torch.cuda.amp.GradScaler(init_scale=131072.0)

    # Dummy input and target
    input = torch.randn(20, 10, device=device)
    target = torch.randn(20, 1, device=device)

    # Loss function
    criterion = nn.MSELoss()

    for step in range(20):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output = model(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Now, check if the model parameters are the same across ranks
        # Gather parameters to rank 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Use CUDA tensors for NCCL backend
                param_cuda = param.detach().clone()
                param_list = [torch.zeros_like(param_cuda) for _ in range(world_size)]
                dist.all_gather(param_list, param_cuda)

                if rank == 0:
                    # Move parameters to CPU for comparison
                    param_list_cpu = [p.cpu() for p in param_list]
                    # Compare the parameters
                    params_equal = True
                    for i in range(1, world_size):
                        if not torch.allclose(param_list_cpu[0], param_list_cpu[i], atol=1e-6):
                            params_equal = False
                            print(f"Step {step}, Parameter '{name}': Divergence detected between rank 0 and rank {i}")
                            print(f"Difference: {torch.norm(param_list_cpu[0] - param_list_cpu[i])}")
                    if params_equal:
                        print(f"Step {step}, Parameter '{name}': Parameters are the same across all ranks.")

    # Clean up
    dist.destroy_process_group()

def main():
    # Get the world_size (number of GPUs)
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()