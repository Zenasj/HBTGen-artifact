import torch
import torch.multiprocessing as mp
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.i = nn.Parameter(torch.tensor(0), requires_grad=False)

def run(rank, net):
    print('start run rank:', rank)

if __name__ == "__main__":
    net = Net()

    net.to('cuda:0')  # Here is the difference from the example in the linked PR

    net.share_memory()
    c = mp.spawn(run, args=[net], nprocs=1, join=False)
    c.join()