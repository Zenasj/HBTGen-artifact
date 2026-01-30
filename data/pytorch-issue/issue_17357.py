import torch
from torch.multiprocessing import Process

def run(rank):
    torch.cuda.set_device(rank)

if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        # it would work fine without the line below
        x = torch.rand(20, 2).cuda()
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()