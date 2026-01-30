import torch
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from tqdm import tqdm


class DummyDataset(Dataset):
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def __getitem__(self, item):
        return item

    def __len__(self):
        return self.n_samples


def main():
    dataset = DummyDataset(20000)
    w = torch.rand(len(dataset)).abs() + 0.1
    dataloader = DataLoader(dataset,
                            sampler=WeightedRandomSampler(w, 65536),
                            num_workers=4,
                            pin_memory=True,
                            batch_size=10,
                            drop_last=True)

    for _ in tqdm(dataloader, total=len(dataloader)):
        pass


if __name__ == '__main__':
    main()