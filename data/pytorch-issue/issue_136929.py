import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._composable.fsdp import fully_shard

def run(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:29500',
        rank=rank,
        world_size=world_size
    )

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 2)

        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            return self.linear2(x)

    model = MyModel().to(device)

    model.linear1.weight.custom_attr = 'custom_value_linear1_weight'
    model.linear1.bias.custom_attr = 'custom_value_linear1_bias'
    model.linear2.weight.custom_attr = 'custom_value_linear2_weight'
    model.linear2.bias.custom_attr = 'custom_value_linear2_bias'

    def check_custom_attrs(model, stage):
        for name, param in model.named_parameters():
            exists = hasattr(param, 'custom_attr')
            value = getattr(param, 'custom_attr', None)
            print(f"Rank {rank} - {stage} - {name}: custom_attr exists? {exists}, value: {value}")

    print(f"\nRank {rank} - Before fully_shard:")
    check_custom_attrs(model, "Before fully_shard")

    fully_shard(model)

    print(f"\nRank {rank} - After fully_shard:")
    check_custom_attrs(model, "After fully_shard")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

attrs = vars(param)
custom_attrs = {k: v for k, v in attrs.items()}

for attr_name, attr_value in custom_attrs.items():
    setattr(self.sharded_param, attr_name, attr_value)