import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# A simple model with 2D weights (Linear layer)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(20, 20)  # 2D weight: (2, 4)
        self.fc2 = nn.Linear(20, 20)  # 2D weight: (2, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def setup_distributed(backend='nccl'):
    # Initialize the process group
    dist.init_process_group(backend)

def cleanup_distributed():
    # Clean up the process group
    dist.destroy_process_group()

def main():
    # Initialize distributed environment
    setup_distributed()
    torch.cuda.set_device(torch.distributed.get_rank())

    # Create the model and move it to GPU
    model = SimpleModel().cuda()

    if torch.cuda.current_device() == 0:
        for n, p in model.named_parameters():
            print(n, p.ndim, p.shape) 
            break
        
    # Wrap the model with FSDP to shard its parameters
    model = FSDP(model, use_orig_params=True)
    
    if torch.cuda.current_device() == 0:
        for n, p in model.module.named_parameters():
            print(n, p.ndim, p.shape)
            break

    # Clean up the distributed environment
    cleanup_distributed()

if __name__ == "__main__":
    main()