import copy
import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.distributed import init_process_group
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW


def build_optimizer(model: nn.Module, paramwise=True):
    base_lr = 1e-4
    if paramwise:
        param_groups = []
        for name, param in model.named_parameters():
            if name.endswith('weight'):
                param_groups.append({'params': [param], 'lr': base_lr * 0.1})
            else:
                param_groups.append({'params': [param], 'lr': base_lr})
        optimizer = AdamW(param_groups, lr=base_lr)
    else:
        optimizer = AdamW(model.parameters(), lr=base_lr)
    return optimizer



class ToyModel(nn.Module):

    def __init__(self, data_preprocessor=None):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.norm = nn.BatchNorm1d(2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        outputs = self.linear1(inputs)
        outputs = self.norm(outputs)
        outputs = self.linear2(outputs)
        return outputs


if __name__ == '__main__':
    init_process_group()
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK')))
    # model: nn.Module = MODELS.build(cfg.model)
    model1 = ToyModel().cuda()
    model2 = copy.deepcopy(model1)
    # train_dataloader = Runner.train_dataloader()
    device_id = torch.cuda.current_device()
    ddp_model = DDP(model1, device_ids=[device_id])
    fsdp_model = FSDP(model2, device_id=device_id, use_orig_params=True, sync_module_states=True)
    ddp_optim_wrapper = build_optimizer(ddp_model)
    fsdp_optim_wrapper = build_optimizer(fsdp_model)
    ddp_scaler = GradScaler()
    fsdp_scaler = GradScaler()
    with autocast():
        for step in range(10):
            data = torch.randn(2, 2).to(f'cuda:{device_id}')
            ddp_loss = ddp_model(data).sum()
            fsdp_loss = fsdp_model(data).sum()

            ddp_scaler.scale(ddp_loss).backward()
            fsdp_scaler.scale(fsdp_loss).backward()

            ddp_scaler.step(ddp_optim_wrapper)
            ddp_scaler.update()
            fsdp_scaler.step(fsdp_optim_wrapper)
            fsdp_scaler.update()

            ddp_optim_wrapper.zero_grad()
            fsdp_optim_wrapper.zero_grad()
            
            print(f'step: {step} rank: {device_id} ddp_loss: {ddp_loss}, fsdp_loss: {fsdp_loss}')