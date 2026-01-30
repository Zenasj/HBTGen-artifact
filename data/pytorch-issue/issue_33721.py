py
import multiprocessing, logging
logger = multiprocessing.log_to_stderr()
logger.setLevel(multiprocessing.SUBDEBUG)

import torch


class Dataset:
    def __len__(self):
        return 23425

    def __getitem__(self, idx):
        return torch.randn(3, 128, 128), idx % 100


ds = Dataset()
trdl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=300, pin_memory=True, shuffle=True)

for e in range(1000):
    for ii, (x, y) in enumerate(trdl):
        print(f'tr {e: 5d} {ii: 5d} avg y={y.mean(dtype=torch.double).item()}')
        if ii % 2 == 0:
            print("="*200 + "BEFORE ERROR" + "="*200)
            1/0