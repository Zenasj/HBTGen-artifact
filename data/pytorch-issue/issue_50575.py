# -*- coding: utf-8 -*-

import argparse
import time

import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import random_split
from torchvision import datasets, transforms
import torch.utils.data.distributed


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# def metric_average(val, name):
#     tensor = torch.tensor(val)
#     avg_tensor = hvd.allreduce(tensor, name=name)
#     return avg_tensor.item()


def run(proc_id, n_gpus, devices, args, kwargs):
    dev_id = devices[proc_id]
    rank = args.nr * n_gpus + proc_id

    world_size = n_gpus * args.nodes

    mnist_train = \
        datasets.MNIST('data-%d' % rank, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    if n_gpus > 1 or args.nodes > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip=args.master_ip, master_port=args.master_port)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=rank)

    train_dataset, test_dataset = random_split(
        mnist_train, [55000, 5000])

    # use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)

    model = Net()
    model = model.to(dev_id)

    if n_gpus > 1 or args.nodes > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id])

    # match ray
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train epoch
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Horovod: set epoch to sampler for shuffling.
        i = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(dev_id), target.to(dev_id)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            i += 1
            if batch_idx % args.log_interval == 0 and rank == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_sampler),
                           100. * batch_idx / len(train_sampler), loss.item()))

        if rank == 0:
            print("batch num is {}".format(i))

    # test
    model.eval()
    test_loss = 0.
    test_accuracy = 0.
    for data, target in test_loader:
        data, target = data.to(dev_id), target.to(dev_id)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        test_accuracy += pred.eq(target.data.view_as(pred)).cpu().float().sum()

    test_loss /= len(test_sampler)
    test_accuracy /= len(test_sampler)

    # print first output on first rank
    # TODO how to get all data's accuracy?
    if rank == 0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
            test_loss, 100. * test_accuracy))

    print("duration time=", time.time() - start)


if __name__ == '__main__':
    start1 = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--nr', type=int, default=0,
                        help="current node's rank")
    parser.add_argument('--gpu', type=str, default='0,1',
                        help="Comma separated list of GPU device IDs.")
    parser.add_argument('--nodes', type=int, default=2,
                        help="number of nodes for training")
    parser.add_argument('--master-ip', type=str, default='10.3.68.117',
                        help="master ip for DDP")
    parser.add_argument('--master-port', type=int, default=12346,
                        help="master port for DDP")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # mnist_train = \
    #     datasets.MNIST('data-0', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ]))

    if n_gpus == 1:
        run(0, n_gpus, devices, args, kwargs)
    else:
        mp.spawn(run, args=(n_gpus, devices, args, kwargs), nprocs=n_gpus)

    print("all time is {} s".format(time.time() - start1))