import torch.nn as nn

import torch

def test1():
    x = torch.Tensor(10, 10)
    x_gpus = torch.cuda.comm.broadcast(x, [2, 3])
    print([t.get_device() for t in x_gpus])

def test2():
    x = torch.Tensor(10, 10).cuda()
    x_gpus = torch.cuda.comm.broadcast(x, [2, 3])
    print([t.get_device() for t in x_gpus])

if __name__ == '__main__':
    test1() # -> [2, 3]
    test2() # -> [0, 3]

# net = torch.nn.DataParallel(net, device_ids=gpus).cuda()
net = torch.nn.DataParallel(net, device_ids=gpus).cuda(device_id=gpus[0])
# ...
# loss_criterion = torch.nn.CrossEntropyLoss().cuda()
loss_criterion = torch.nn.CrossEntropyLoss().cuda(device_id=gpus[0])
# ...
# input = input.cuda()
input = input.cuda(device=gpus[0])
# target = target.cuda()
target = target.cuda(device=gpus[0])