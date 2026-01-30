import torch.nn as nn

while True:
        net = torch.nn.Linear(5000,5000)
        net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer,
                'max', patience=args.patience, factor=0.5, verbose=True)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30,
        #         gamma=0.1)

import torch

while True:
        net = torch.nn.Linear(5000,5000)
        net = net.cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, 'max', patience=20, factor=0.5, verbose=True)

import torch
from pytorch_transformers import WarmupConstantSchedule, WarmupCosineSchedule, WarmupLinearSchedule, WarmupCosineWithHardRestartsSchedule

# In my actual project, this is a for loop over the k-folds of k-fold cross-validation.
# In this example I use a while just to demonstrate the OOM error.
while True:
    net = torch.nn.Linear(10000, 10000)
    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # scheduler = WarmupCosineWithHardRestartsSchedule(optimizer, 1, 1000)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    
    
    # I also tried all these other schedulers. Same issue.
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,80])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    # scheduler = WarmupConstantSchedule(optimizer, 1)
    # scheduler = WarmupCosineSchedule(optimizer, 1, 1000)
    # scheduler = WarmupLinearSchedule(optimizer, 1, 1000)

    del net, optimizer, scheduler

import torch

while True:
    net = torch.nn.Linear(10000, 10000)
    net = net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    del net, optimizer