import os
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed

# Error occurs when dataset_size * 2 < num_gpus
num_gpus = 9
dataset_size = 4

def main(gpu):
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=num_gpus, rank=gpu)

    dataset = DummyDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_gpus, rank=gpu)
    loader = torch.utils.data.DataLoader(dataset, 1, sampler=sampler)

    for batch in loader:
        print(batch)

class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, item):
        return 0
    def __len__(self):
        return dataset_size

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '10000'
    mp.spawn(main, nprocs=num_gpus)