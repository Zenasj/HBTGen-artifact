import torch
from torch.multiprocessing import Process

@torch.jit.script
def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))


class Net(torch.jit.ScriptModule):
    def __init__(self, n):
        super().__init__()

    @torch.jit.script_method
    def forward(self, x):
        return torch.sigmoid(x)
        # return sigmoid(x)
        # return torch.tanh(x)

def run(rank,):
    xs = torch.randn((5, 3), dtype=torch.float32)
    f = Net(3)
    print("{}: Starting...".format(rank))
    print(f.forward(xs))
    print("{}: Done!".format(rank))

if __name__ == "__main__":
    # These lines are important
    xs = torch.randn((5, 3), dtype=torch.float32)
    net = Net(3)
    print(net(xs))

    processes = []
    for rank in range(2):
        p = Process(target=run, args=(rank,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()