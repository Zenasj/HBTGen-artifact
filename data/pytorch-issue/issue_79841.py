import random

import torch.nn as nn
import torch
import numpy as np
import torch.multiprocessing as mp
import os
import torch.distributed as dist

class R2D2(nn.Module):
    def __init__(self):
        super(R2D2, self).__init__()
        self.conv=nn.Conv2d(4, 16, kernel_size=3,stride=4)
        self.core=nn.LSTM(input_size=16*21*21,hidden_size=512,batch_first=True)
        self.head = nn.Linear(512,15)


    def forward(self,state,hidden_state):
        B=state.shape[0]
        T=state.shape[1]
        state=state.flatten(start_dim=0, end_dim=1)
        latent=self.conv(state)
        latent=latent.reshape(B,T,-1)
        x,hidden_state=self.core(latent,hidden_state)
        x=self.head(x.flatten(start_dim=0, end_dim=1)).reshape(B,T,-1)
        return x,hidden_state


def run(rank,size):
    autocast=True
    torch.cuda.set_device(rank)
    burnin=40
    model=R2D2().cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                      output_device=rank,find_unused_parameters=True)
    scaler=torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    target=torch.rand(64,80,1, dtype=torch.float32).detach().cuda()
    for i in range(1):
        with torch.cuda.amp.autocast(enabled=autocast):
            optimizer.zero_grad(set_to_none=True)
            state=torch.from_numpy(np.random.randint(0, 255, (64,120, 4, 84, 84), dtype=np.uint8)).pin_memory().cuda()
            state=state/255.0
            hidden_state = (torch.rand((1, 64, 512), dtype=torch.float32).pin_memory().cuda(),
                            torch.rand((1, 64, 512), dtype=torch.float32).pin_memory().cuda())
            with torch.no_grad():
                _,hidden_state=model(state[:,0:burnin],hidden_state)

            x,hidden_state=model(state[:,burnin:],hidden_state)
            loss=torch.mean((x-target)**2)
        if autocast:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=size)
    fn(rank, size)


if __name__ == '__main__':
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()