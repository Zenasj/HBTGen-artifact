import torch
print("pytorch version",torch.__version__) 
import torch.nn as nn
model = nn.Linear(1, 1) # 'Net' is a simple MLP
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [4,7], gamma=0.1)

print('Initial LR : {0:.8f}'.format(schedular.get_lr()[0]))
for e in range(8):
  optimizer.step()
  schedular.step()
  print('Epoch {0}, LR: {1:.8f}, opt LR {2:.8f}'.format(e, schedular.get_last_lr()[0],
          optimizer.param_groups[0]['lr']))