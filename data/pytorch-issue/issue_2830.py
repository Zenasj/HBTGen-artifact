import torch

optimizer = optim.Adam()
optimizer.load_state_dict(checkpoint['optimizer'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

model = Model()
model.load_state_dict(checkpoint['model'])
model.cuda()
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

model = Model()
model.cuda()
optimizer = optim.Adam(model.parameters())

for d, gt in trn_dataloader:
    # train
    ... 
    optimizer.step()
    model.cpu() # move to cpu
    # eval or do other things
    ...
    model.cuda()  # but finnally, move back