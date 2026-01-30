import torch.nn as nn

import torch
import time
class BugModule(torch.nn.Module):
    def __init__(self, num_mod):
        torch.nn.Module.__init__(self)
        self.modlist = torch.nn.ModuleList(
            [torch.nn.Linear(1000, 50) for _ in range(num_mod)])

    def forward(self, x):
        out = self.modlist[0](x)
        return out


model2 = BugModule(2)
model200 = BugModule(200)
model2 = model2.cuda()
model200 = model200.cuda()
model2 = torch.nn.DataParallel(model2)
model200 = torch.nn.DataParallel(model200)

model2_times = []
model200_times = []
out = model2(torch.FloatTensor(50, 1000))
out = model200(torch.FloatTensor(50, 1000))
for i in range(200):
    t = time.time()
    out = model2(torch.FloatTensor(50, 1000))
    model2_times.append(time.time() - t)

for i in range(200):
    t = time.time()
    out = model200(torch.FloatTensor(50, 1000))
    model200_times.append(time.time() - t)

print('Model 2 takes {} sec'.format(sum(model2_times) / 200.0))
print('Model 200 takes {} sec'.format(sum(model200_times) / 200.0))