import random
import torch
from torch.utils.data import DataLoader, Dataset


class RandomDatasetMock(Dataset):
    def __getitem__(self, index):
        return torch.tensor([torch.rand(1).item(), random.uniform(0, 1)])

    def __len__(self):
        return 1000


dataloader = DataLoader(RandomDatasetMock(),
                        batch_size=2,
                        num_workers=1,
                        drop_last=False)

reveal_type(dataloader)