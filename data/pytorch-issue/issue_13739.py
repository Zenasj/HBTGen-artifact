import os
import time

import torch


def check():
    indices = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
    values = torch.FloatTensor([3, 4, 5])
    tensor = torch.sparse_coo_tensor(indices, values, torch.Size([2,4]))


def main():
    check()  # Note: this line is required to reproduce behavior!
    pid = os.fork()
    if pid:
        print(f"Starting check pid: {pid}")
        check()
        print(f"Done check pid: {pid}")
        time.sleep(10)
    else:
        print(f"Starting check pid: {pid}")
        check()
        print(f"Done check pid: {pid}")
        time.sleep(10)


if __name__ == '__main__':
    main()