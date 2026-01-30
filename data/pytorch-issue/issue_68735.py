import torch

# DISTRIBUTED: this is required to initialize the pytorch backend
dist_url = "env://" # default
# dist_url = "auto"
dist.init_process_group(
    backend=distributed_backend,
    init_method=dist_url,
    timeout=datetime.timedelta(seconds=2000), #default is 30 min
    world_size=world_size,
    rank=world_rank  
)


torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank) if torch.cuda.is_available() else 'cpu'
print("device is {}".format(device))
# this will make all .cuda() calls work properly

# synchronize all the threads to reach this point before moving on
dist.barrier()