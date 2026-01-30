import torch
import torch.nn as nn

torch.manual_seed(2809)

# Creates model and optimizer in default precision
model = nn.Linear(10, 10).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1., fused=False)

# Creates a GradScaler once at the beginning of training.
scaler = torch.cuda.amp.GradScaler()

data = torch.randn(10, 10, device='cuda')
target = torch.randn(10, 10, device='cuda')
loss_fn = nn.MSELoss()

epochs = 5
for epoch in range(epochs):
    optimizer.zero_grad()
    # Runs the forward pass with autocasting.
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(data)
        loss = loss_fn(output, target)

    # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
    # Backward passes under autocast are not recommended.
    # Backward ops run in the same dtype autocast chose for corresponding forward ops.
    scaler.scale(loss).backward()

    # scaler.step() first unscales the gradients of the optimizer's assigned params.
    # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
    # otherwise, optimizer.step() is skipped.
    scaler.step(optimizer)

    # Updates the scale for next iteration.
    scaler.update()
    
    print("epoch {}, loss {:.3f}".format(epoch, loss.item()))


# fused=False
# epoch 0, loss 1.302
# epoch 1, loss 11.289
# epoch 2, loss 2.388
# epoch 3, loss 2.994
# epoch 4, loss 5.163

# fused=True
# epoch 0, loss 1.302
# epoch 1, loss 11.289
# epoch 2, loss 2.388
# epoch 3, loss 2.994
# epoch 4, loss 5.163