import multiprocessing, logging

mpl = multiprocessing.log_to_stderr()
mpl.setLevel(logging.INFO)

import torch
import tqdm


class TrivialDataset(torch.utils.data.Dataset):

    def __init__(self, num_samples, sample_size):
        self.num_samples = num_samples
        self.sample_size = sample_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        return torch.zeros((self.sample_size, 1),
                           dtype=torch.float32,
                           device='cpu')


if __name__ == "__main__":

    torch.set_num_threads(1)
    dataloader = torch.utils.data.DataLoader(
        dataset=TrivialDataset(10000, 16384),
        num_workers=8,
    )

    for i in tqdm.tqdm(dataloader):
        pass