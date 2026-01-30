import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.autograd import Variable


def task(pid, model):
    x = Variable(torch.rand(64, 10))
    y = model(x)
    t = y.clone() * 0.99
    loss = F.smooth_l1_loss(y, t)

    # here it breaks
    loss.backward()

    print("Process %d finished" % pid)


if __name__ == "__main__":

    # comment manual_seed and the CUDA initialization error is gone.
    torch.manual_seed(23)

    net = nn.Linear(10, 4)
    net.share_memory()

    processes = []
    for pid in range(8):
        p = mp.Process(target=task, args=(pid, net))
        p.start()

    for p in processes:
        p.join()

    print("Done.")

for i, (data, labels) in enumerate(train_loader):
    pass

for i, (data, labels) in enumerate(train_loader):
    pass