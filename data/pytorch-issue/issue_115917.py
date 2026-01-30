from threading import Thread

import torch

def task1():
    torch.set_default_device('cuda')
    print(torch.tensor([1, 2, 3]).device)


if __name__ == '__main__':
    torch.set_default_device('cpu')
    print(torch.tensor([1, 2, 3]).device)

    t1 = Thread(target=task1)
    t1.start()
    t1.join()