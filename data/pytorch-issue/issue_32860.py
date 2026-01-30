python
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist


def dp_warning(batch_size, gpu_ids, use_flatten=True):
    emb_sz = 100
    seq_len = 128
    # batch x seq_len x emb_size
    bilstm = nn.LSTM(
        input_size=emb_sz,
        hidden_size=100,
        num_layers=2,
        batch_first=True,
        dropout=0.1,
        bidirectional=True)

    device = torch.device("cuda:" + str(gpu_ids[0]))
    model = DataParallel(bilstm.cuda(device), device_ids=gpu_ids)
    data = torch.rand([batch_size, seq_len, emb_sz]).cuda(device)

    niter = 10
    model.train()
    for _ in range(niter):
        if use_flatten:
            bilstm.flatten_parameters()
        model(data)


def distributed_warning(gpu, batch_size, world_size, use_flatten=True):
    emb_sz = 100
    seq_len = 128

    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend="nccl", rank=gpu, world_size=world_size)

    # batch x seq_len x emb_size
    bilstm = nn.LSTM(
        input_size=emb_sz,
        hidden_size=100,
        num_layers=2,
        batch_first=True,
        dropout=0.1,
        bidirectional=True)

    model = DistributedDataParallel(bilstm.cuda(gpu), device_ids=[gpu])
    data = torch.rand([batch_size, seq_len, emb_sz]).cuda(gpu)
    niter = 10
    model.train()
    for _ in range(niter):
        if use_flatten:
            bilstm.flatten_parameters()
        model(data)

    dist.destroy_process_group()


def distributed_main(batch_size, gpu_ids, use_flatten=True):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5555'
    mp.spawn(
        distributed_warning,
        nprocs=len(gpu_ids),
        args=(batch_size, len(gpu_ids), use_flatten))


if __name__ == '__main__':
    # dp_warning(10, [0, 1], use_flatten=True)
    distributed_main(10, [0, 1], use_flatten=True)