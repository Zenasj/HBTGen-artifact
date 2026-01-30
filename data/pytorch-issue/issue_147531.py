import torch.nn as nn

import torch
import torch.distributed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import os

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0]))
        self.batch_size = 128
    def forward(self):
        return torch.randn(self.batch_size) * self.a

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.distributed.init_process_group()
    device = f'cuda:{local_rank}'
    model = Model().to(device)
    model = FSDP(model)
    print(model())

if __name__ == '__main__':
    main()