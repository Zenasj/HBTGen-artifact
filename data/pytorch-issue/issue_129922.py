import torch.distributed as dist
import torch
import torch.nn as nn
import os
import torch, torch.nn as nn, torch.optim as optim
from apex.contrib.optimizers.distributed_fused_adam import DistributedFusedAdam
import copy

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(global_rank)
    learning_rate = 0.0001
    decay = 0
    num_iters = 30
    amp = True

    dist.init_process_group(backend='nccl', init_method='env://')
    world_group = torch.distributed.new_group(ranks=[i for i in range(world_size)])
    self_groups = [torch.distributed.new_group(ranks=[i]) for i in range(world_size)]

    model = ToyModel().to(device).half()
    loss_fn = nn.CrossEntropyLoss().to(device)
    

    optimizer = DistributedFusedAdam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=decay,
        overlap_grad_sync=False,
        contiguous_param_buffer=True,
        contiguous_grad_buffer=True,
        distributed_process_group=self_groups[global_rank],
        redundant_process_group=world_group,
        grad_sync_dtype=torch.float16 if amp else torch.float32,
        param_sync_dtype=torch.float16 if amp else torch.float32
    )
  
    grad_scaler = torch.amp.GradScaler('cuda', init_scale=1024)
    
    sched = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10)
    input = torch.randn(20, 10).to(device)
    input = input.half()
    labels = torch.ones(20, 5).to(device)

    for i in range(num_iters):
        optimizer.zero_grad()
        outputs = model(input)
        loss = loss_fn(outputs, labels)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

if __name__ == "__main__":
    main()

[tasklist]
### Tasks