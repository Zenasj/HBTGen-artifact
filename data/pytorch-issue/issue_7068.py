import torch.nn as nn
import torchvision

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0)),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=0)

network = model.Cifar()
network.cuda()

import sys
import random
import datetime as dt

import numpy as np
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True

features = torch.randn(2, 5)

# Print stuff.
fnp = features.view(-1).numpy()

print("Time: {}".format(dt.datetime.now()))
for el in fnp:
    print("{:.20f}".format(el))

print("Python: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Pytorch: {}".format(torch.__version__))

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,   pin_memory=True, worker_init_fn=_init_fn)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False