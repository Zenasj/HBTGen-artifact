import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return 60000 * 64
    def __getitem__(self, idx):
        data_1 = np.zeros((50, 2))
        return {"d1": data_1}


if __name__ == "__main__":
    dl = DataLoader(MyDataset(), batch_size=64, num_workers=4, shuffle=False)
    dt_1 = []

    for data in dl:
        dt_1.append(data["d1"].numpy())