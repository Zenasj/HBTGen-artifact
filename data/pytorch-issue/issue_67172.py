def init(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

def debug_torch():
    world_size = 4
    os.environ['MASTER_ADDR'] = 'localhost'  #
    os.environ['MASTER_PORT'] = '123456'  #
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    mp.spawn(init, nprocs=world_size, args=(world_size,))