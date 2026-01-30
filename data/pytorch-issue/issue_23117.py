from torch import multiprocessing
from torch.utils.data import DataLoader
import torch

mp_lock = multiprocessing.Lock()

def main(device_index):
    list(DataLoader([torch.tensor(i) for i in range(10)], num_workers=4))


if __name__ == '__main__':
    torch.multiprocessing.spawn(main)