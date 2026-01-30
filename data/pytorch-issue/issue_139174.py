import torch.nn as nn

import torch
N, D_in, D_out = 64, 1024, 16
x = torch.randn(N, D_in, device='cuda')
y = torch.randn(N, D_out, device='cuda')

model = torch.nn.Linear(D_in, D_out).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
loss_fn = torch.nn.MSELoss()

# from torch.cuda.amp import GradScaler, autocast

from gradscaler2 import GradScaler
scaler = GradScaler()

def run_fwd_bwd():
    with torch.cuda.amp.autocast():
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    optimizer.zero_grad(set_to_none=True)
    scaler.update()

for t in range(20):
    run_fwd_bwd()