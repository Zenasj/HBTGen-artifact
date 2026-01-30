import torch
import torchvision
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp

import os
from torchvision import transforms
from torchvision.transforms import RandomResizedCrop
print(torchvision.__version__)
print(torch.__version__)
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      RandomResizedCrop(224),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])

train_data = torchvision.datasets.MNIST('data', train=True, transform=mnist_transform, download=True)
# train_data = torchvision.datasets.CIFAR10('data', train=True,download=True, transform=mnist_transform)

def main_fun(rank, world_size):
    print('rank=',rank)
    print('world_size=', world_size)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist_backend = 'nccl'
    dist_url = 'env://'

    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            world_size=world_size, rank=rank)
    dist.barrier()

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size=16, drop_last=True)

    print(len(train_data) , len(train_batch_sampler), train_batch_sampler)

    for i in train_sampler: # train_batch_sampler
        print(type(i),i)

    print('finished...')

def main():
    world_size = 2
    # mp.spawn(main_fun,
    #          args=(world_size,),
    #          nprocs=world_size,
    #          join=True)

    processes = []
    for rank in range(world_size):
        p = Process(target=main_fun, args=(rank, world_size))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()

train_data = torchvision.datasets.MNIST('data', train=False, transform=mnist_transform, download=True)