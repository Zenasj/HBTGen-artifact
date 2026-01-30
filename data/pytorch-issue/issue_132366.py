import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import torch.distributed.checkpoint.state_dict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.small_linear = nn.Linear(1,1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        zero = torch.zeros(1, x.device)
        return output + self.small_linear.forward(zero)


if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = Net().to(rank)
    model = FSDP(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    module_state_dict, optimizer_state_dict = torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizer)
    state_dict = {
        "module": module_state_dict,
        "optimizer": optimizer_state_dict
    }
    torch.distributed.checkpoint.save(state_dict, checkpoint_id="temp/checkpoint")

    dist.destroy_process_group()

self.scalar

self.scalar = nn.Linear(1,1)

self.scalar = nn.Parameter(torch.Tensor([1.0, 2.0]))

self.scalar = nn.Linear(100,100)

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import torch.distributed.checkpoint.state_dict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.small_linear = nn.Linear(1,1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        zero = torch.zeros(1, x.device)
        return output + self.small_linear.forward(zero)


if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    from torch.distributed.device_mesh import init_device_mesh

    device_mesh = init_device_mesh("cuda", (world_size,))

    model = Net().to(rank)
    # pass device_mesh to FSDP so you can get DTensor state_dict
    model = FSDP(model, device_mesh=device_mesh)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    module_state_dict, optimizer_state_dict = torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizer)
    state_dict = {
        "module": module_state_dict,
        "optimizer": optimizer_state_dict
    }
    torch.distributed.checkpoint.save(state_dict, checkpoint_id="temp/checkpoint")

    dist.destroy_process_group()

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint
import torch.distributed.checkpoint.state_dict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.device_mesh import init_device_mesh


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.scalar = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output + self.scalar        


if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    device_mesh = init_device_mesh("cuda", (world_size,))

    model = Net().to(rank)
    model = FSDP(model, device_mesh=device_mesh)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    module_state_dict, optimizer_state_dict = torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizer)
    state_dict = {
        "module": module_state_dict,
        "optimizer": optimizer_state_dict
    }    
    torch.distributed.checkpoint.load(state_dict, checkpoint_id="temp/checkpoint")
    torch.distributed.checkpoint.state_dict.set_state_dict(
        model, 
        optimizer, 
        model_state_dict=state_dict["module"], 
        optim_state_dict=state_dict["optimizer"])

    dist.destroy_process_group()

test.py

test_load.py

self.scalar