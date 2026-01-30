import torch

sched = torch.optim.lr_scheduler.StepLR(opt, 2, 2.)  # step size = 2, gamma = 2
for i in range(5):
    opt.step()
    opt.step()
    sched.step(epoch=2)  # fix the epoch
    print(opt.state_dict()['param_groups'][0]['lr'])

2.0
4.0
8.0
16.0
32.0

sched = torch.optim.lr_scheduler.StepLR(opt, 2, 2.)  # step size = 2, gamma = 2
for i in range(5):
    opt.step()
    opt.step()
    sched.step(epoch=3)  # fix the epoch
    print(opt.state_dict()['param_groups'][0]['lr'])
1.0
1.0
1.0
1.0
1.0