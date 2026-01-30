import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

ctx = mp.get_context("spawn")

class RandomSet(Dataset):

    def __getitem__(self, item):
        return torch.randn(256,256,256)

    def __len__(self):
        return 1000


class CatModel(nn.Module):
    def __init__(self, use_cat=True):
        super(CatModel, self).__init__()

        self.use_cat = use_cat

        if use_cat:
            self.conv = nn.Conv2d(512, 512, 3)
        else:
            self.conv = nn.Conv2d(256, 512, 3)

    def forward(self, x):

        if self.use_cat:
            x = torch.cat([x, x], dim=1)

        return self.conv(x)


# change either one to False: No additional CUDA context created
use_cat = True
pin_memory = True

class Trainer(ctx.Process):
    def __init__(self, device):
        super(Trainer, self).__init__()

        self.device = device

        self.dataset = RandomSet()
        self.dataloder = DataLoader(
            self.dataset,
            pin_memory=pin_memory,
            batch_size=8,
            num_workers=8
        )

    def run(self):
        self.model = CatModel(use_cat).to(self.device)

        # loop
        for x in self.dataloder:
            x = x.to(self.device)
            y = self.model(x)


if __name__ == "__main__":

    trainer1 = Trainer(0)
    trainer2 = Trainer(1)

    trainer1.start()
    trainer2.start()

    trainer1.join()
    trainer2.join()