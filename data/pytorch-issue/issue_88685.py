import torch

def run(rank, world_size, data,  dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl',  rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

    if rank == 0 :
        objects = ["foo", 12, {1: 2}] # any picklable object
    else:
        objects = [None, None, None]
    objects =objects
    outputlist = [None]
    dist.scatter_object_list(outputlist, objects, src=0)
    print(outputlist)


if __name__ == '__main__':
    dataset = Reddit('/data/Reddit')
    world_size = torch.cuda.device_count()
    data = dataset[0]

    print('Let\'s use', world_size, 'GPUs!')
    data_split = (data.train_mask, data.val_mask, data.test_mask)
    mp.spawn(
        run,
        args=(world_size, data, dataset),
        nprocs=world_size,
        join=True
    )