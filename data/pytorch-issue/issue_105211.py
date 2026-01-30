import torch
import torch.nn as nn
import torch.nn.functional as F

l1 = nn.Linear(2, 2).cuda()
l2 = nn.Linear(2, 2).cuda()
l2.load_state_dict(l1.state_dict())

x = torch.randn(2, 2).cuda()
y = torch.randn(2, 2).cuda()

##################################
# Here l1_loss does not have a grad_fn
##################################
with torch.autocast(device_type='cuda', dtype=torch.float16):
    with torch.no_grad():
        xhat = l1(x)
    yhat = l1(xhat)
    l1_loss = F.mse_loss(yhat, y)
l1_loss.backward()

##################################
# Here l2_loss DOES have a grad_fn
##################################
with torch.autocast(device_type='cuda', dtype=torch.float16):
    xhat = l2(x).detach()
    yhat = l2(xhat)
    l2_loss = F.mse_loss(yhat, y)
l2_loss.backward()