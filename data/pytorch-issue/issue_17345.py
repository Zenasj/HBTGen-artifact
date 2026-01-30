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
    net.share_memory()
    c = mp.spawn(run, args=[net], nprocs=1, join=False)
    c.join()

net = Net()
net.share_memory()
state_dict = copy.deepcopy(net.state_dict())
c = mp.spawn(run, args=[state_dict], nprocs=1, join=False)
...
# later times
state_dict.update(copy.deepcopy(net.state_dict()))
...
# in run(state_dict):
local_net.load_state_dict(state_dict)

state_dict = {
    'state': copy.deepcopy(net.state_dict())
}
# later times
state_dict['state'] = copy.deepcopy(net.state_dict()) # instead of .update(...)