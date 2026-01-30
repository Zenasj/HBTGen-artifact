import torch
import torch.nn as nn

model = BidirectionalLSTM(10, 10, 10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
input = torch.randn(16, 10, 10, device='cuda')
loss_fn = nn.CrossEntropyLoss()
target = torch.randint(0, 10, (16, 10), device='cuda')
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()