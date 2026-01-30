"""
For code used in distributed training.
"""
from typing import Tuple

import torch
import torch.distributed as dist

import os

from torch import Tensor

import torch.multiprocessing as mp

def set_sharing_strategy(new_strategy=None):
    """
    https://pytorch.org/docs/stable/multiprocessing.html
    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    https://stackoverflow.com/questions/66426199/how-does-one-setup-the-set-sharing-strategy-strategy-for-multiprocessing-in-pyto
    """
    from sys import platform

    if new_strategy is not None:
        mp.set_sharing_strategy(new_strategy=new_strategy)
    else:
        if platform == 'darwin':  # OS X
            # only sharing strategy available at OS X
            mp.set_sharing_strategy('file_system')
        else:
            # ulimit -n 32767 or ulimit -n unlimited (perhaps later do try catch to execute this increase fd limit)
            mp.set_sharing_strategy('file_descriptor')

def use_file_system_sharing_strategy():
    """
    when to many file descriptor error happens

    https://discuss.pytorch.org/t/how-does-one-setp-up-the-set-sharing-strategy-strategy-for-multiprocessing/113302
    """
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def setup_process(rank, world_size, port, backend='gloo'):
    """
    Initialize the distributed environment (for each process).

    gloo: is a collective communications library (https://github.com/facebookincubator/gloo). My understanding is that
    it's a library/API for process to communicate/coordinate with each other/master. It's a backend library.

    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_DISABLE=1

    https://stackoverflow.com/questions/61075390/about-pytorch-nccl-error-unhandled-system-error-nccl-version-2-4-8

    https://pytorch.org/docs/stable/distributed.html#common-environment-variables
    """
    import torch.distributed as dist
    import os
    import torch

    if rank != -1:  # -1 rank indicates serial code
        print(f'setting up rank={rank} (with world_size={world_size})')
        # MASTER_ADDR = 'localhost'
        MASTER_ADDR = '127.0.0.1'
        # set up the master's ip address so this child process can coordinate
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        print(f"{MASTER_ADDR=}")
        os.environ['MASTER_PORT'] = port
        print(f"{port=}")

        # - use NCCL if you are using gpus: https://pytorch.org/tutorials/intermediate/dist_tuto.html#communication-backends
        if torch.cuda.is_available():
            # unsure if this is really needed
            # os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
            # os.environ['NCCL_IB_DISABLE'] = '1'
            backend = 'nccl'
        print(f'{backend=}')
        # Initializes the default distributed process group, and this will also initialize the distributed package.
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend, rank=rank, world_size=world_size)
        # dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        print(f'--> done setting up rank={rank}')

def cleanup(rank):
    """ Destroy a given process group, and deinitialize the distributed package """
    # only destroy the process distributed group if the code is not running serially
    if rank != -1:  # -1 rank indicates serial code
        dist.destroy_process_group()

def get_batch(batch: Tuple[Tensor, Tensor], rank) -> Tuple[Tensor, Tensor]:
    x, y = batch
    if torch.cuda.is_available():
        x, y = x.to(rank), y.to(rank)
    else:
        # I don't think this is needed...
        # x, y = x.share_memory_(), y.share_memory_()
        pass
    return x, y

def test_setup():
    print('test_setup')
    port = find_free_port()
    world_size = 4
    mp.spawn(setup_process, args=(world_size, port), nprocs=4)
    print('successful test_setup!')


if __name__ == '__main__':
    test_setup()

# python3 DDP.py --projectName "PyTorch-4K-2X" --batchSize 32 --nEpochs 2 --lr 1e-3 --step 10 --threads 8 --optim AdamW

import argparse, os, sys
import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ... import Net
from ... import loss
from ... import CustomDataset
import metrics

def main():
    
    parser = argparse.ArgumentParser(description="PyTorch-W&B-Training")
    
    parser.add_argument("--projectName", default="PyTorch-4K-2X", type=str, help="Project Name for W&B")
    parser.add_argument("--batchSize", type=int, default=12, help="Training batch size")
    parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
    parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs,   Default: n=10")
    parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
    parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument("--clip", type=float, default=0.1, help="Clipping Gradients. Default=0.1")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads for data loader to use, Default: 4")
    parser.add_argument("--optim", default='AdamW', type=str, help="AdamW optimizer")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
    parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')

    #parser.add_argument("--rank", default="4", type=str, help="rank (default: 0)")
    parser.add_argument('-g', '--gpus', default=4, type=int,
                            help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                            help='ranking within the nodes')
    parser.add_argument("--world_size", default=4, type=int, help="world_size = num_gpus that you want to train upon (default: 4)")

    parser.add_argument("--trainDir", type=str, default='/opt/hubshare/vectorly-share/shared/Image_Superresolution/Dataset/Train/CombinedALL/2X/', help="Training Dataset Path") # Train/CombinedALL/2X/
    parser.add_argument("--trainInputSize", type=int, default=256, help="Training Data Input Image Size")
    
    
    cudnn.benchmark = True  
    torch.backends.cudnn.determinstic = False
    random.seed(hash("setting random seeds") % 2**32 - 1)
    np.random.seed(hash("improves reproducibility") % 2**32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    
    import wandb
    wandb.login()
    
    global opt
    opt = parser.parse_args()
    print(opt)
    
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    
    
    mp.spawn(train, nprocs=opt.world_size, args=(opt, ))
    
    
def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_process(rank, master_addr, master_port, world_size, backend='nccl'):
    print(f'setting up {rank} {world_size} {backend}')

    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"{master_addr} {master_port}")

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"{rank} init complete")
    dist.destroy_process_group()
    print(f"{rank} destroy complete")
    
    
def train(gpu, opt):
    
    print("===> Setting Environment for DDP")
    
    rank = opt.nr * opt.gpus + gpu
    print(f"Running DDP based training on rank {rank}.")
    
#     dist.init_process_group(
#         backend='nccl',
#         #init_method='env://',
#         world_size=opt.world_size,
#         rank=rank)

    master_addr='10.1.1.20'
    master_port = find_free_port()
    
    setup_process(rank, master_addr, master_port, opt.world_size, backend='nccl')
    
    print("===> Environment successfully configured for DDP")
    dist.destroy_process_group()

    epochs = opt.nEpochs
    project_name = opt.projectName
    
    config = dict(
        epochs=epochs,
        batch_size=opt.batchSize,
        learning_rate=opt.lr,
        dataset="DemoVal",
        architecture="4K-2X"
    )
    
    
    
if __name__ == "__main__":
    main()