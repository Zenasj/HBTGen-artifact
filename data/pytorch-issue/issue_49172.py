import torch

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, length: int):
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx: int):
        return torch.rand(100), torch.rand(5)

num_replicas = 7    # 6 works fine
for i in range(num_replicas):
    sampler = torch.utils.data.DistributedSampler(
        RandomDataset(3), num_replicas=num_replicas, rank=i
    )
    print(list(sampler))