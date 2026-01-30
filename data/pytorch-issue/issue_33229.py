import torch
t = torch.tensor(1.0, requires_grad=True)
opt = torch.optim.SGD([t], lr=0.01)
s = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[19])
print('lr', opt.param_groups[0]['lr'])

opt.step()
s.step(0)
print('lr', opt.param_groups[0]['lr'])