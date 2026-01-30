import torch

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def init_process(rank, world_size):
    # IP address of machine on which process 0 is located
    # free port on the machine on which process 0 is located
    os.environ['MASTER_ADDR'] = Master_IP
    os.environ['MASTER_PORT'] = find_free_port()
    dist.init_process_group(
        backend="nccl", init_method='env://', world_size=world_size, rank=rank)
    dist.barrier()

def setup_train_loader(train_dataset, train_sampler, world_size, rank):

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size/world_size),
        num_workers=cfg.num_worker,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
    )
    return train_loader

def setup_val_loader(val_dataset, val_sampler, world_size, rank):
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.batch_size/world_size),
        num_workers= cfg.num_worker,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        drop_last=True,
        pin_memory=True,
    )
    return val_loader

def train(rank):

    world_size = torch.cuda.device_count()

    init_process(rank, world_size)
    if dist.is_initialized():
        print(f"Rank {rank + 1}/{world_size} process initialized.\n")
    else:
        sys.exit()

    torch.manual_seed(0)
    torch.cuda.set_device(rank)

    print('Getting dataset')
    train_dataset = Generator(cfg, mode='Train')
    val_dataset = Generator(cfg, mode='Validate')

    # setting up model
    Model = PINet()
    # setting up optimizer
    # setting up scheduler

    print('Setting up dataloader')
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None

    train_loader = setup_train_loader(train_dataset, train_sampler, world_size, rank)

    if dist.is_initialized():
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        val_sampler = None

    val_loader = setup_val_loader(val_dataset, val_sampler, world_size, rank)

    start_time = time.time()
    for epoch in range(cfg.n_epoch):
        Model.training_mode()
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

            for t_batch, sample in enumerate(train_loader):
                imgs = sample['imgs']
                labels = sample['labels']
                # regular training loop



if __name__ == '__main__':
    # training()
    # considering 1 machine and N GPUs. for multiple machines and multiple GPUs, this process gets complicated.
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        mp.spawn(training, nprocs=world_size, join=True)

train_dataset = Generator(cfg, mode='Train')

self.cfg = cfg
self.actual_batchsize = None
self.mode = mode
self.dataset_size = None
self.train_data = []
self.test_data = []
self.val_data = []
self.process_data = ProcessData(cfg)