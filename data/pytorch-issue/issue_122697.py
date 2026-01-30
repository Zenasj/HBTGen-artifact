import torch.nn as nn

import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)

set_seed(42)
eval_testset = True

for epoch in range(10):
    if epoch >= 2:
        break # for the purpose of saving time

    for i, data in enumerate(trainloader):
        inputs, labels = data
        # train the model ...
        if i == 0:
            print("eval_testset=%s, epoch=%d, iter=%d, loaded labels=%s"%(eval_testset, epoch, i, labels.tolist()))

    if eval_testset:
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # test the model ...

import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms

torch.use_deterministic_algorithms(True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)

for eval_testset in [False, True]:
    print('-------------------')
    print(f'eval_testset: {eval_testset}')
    print('-------------------')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_g = torch.Generator()
    test_g = torch.Generator()
    train_g.manual_seed(102)
    test_g.manual_seed(234)

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=train_g)

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        worker_init_fn=seed_worker,
        generator=test_g)

    set_seed(42)

    for epoch in range(10):
        if epoch >= 2:
            break # for the purpose of saving time

        for i, data in enumerate(trainloader):
            inputs, labels = data
            # train the model ...
            if i == 0:
                print("eval_testset=%s, epoch=%d, iter=%d, loaded labels=%s"%(eval_testset, epoch, i, labels.tolist()))

        if eval_testset:
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    # test the model ...

import torch

torch.manual_seed(0)
print(torch.randint(0, 200, (1, )).item())

torch.manual_seed(0)
data = torch.arange(100)

dl = torch.utils.data.DataLoader(
    data, batch_size=4, shuffle=False, generator=torch.Generator())

iter(dl)
print(torch.randint(0, 200, (1, )).item())