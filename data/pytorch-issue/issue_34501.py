import torch

# forward pass has not RPC
loss = ...

# thread 1
loss.backward()

# thread 2
with dist_autograd.context() as context_id:
  dist_autograd.backward(context_id, [loss])

# thread 1
with dist_autograd.context() as context_id:
  with ddp.no_sync():
    for _ in range(5):
      loss = ddp(input).sum()
      dist_autograd.backward(context_id, [loss])
  loss = ddp(input).sum()
  dist_autograd.backward(context_id, [loss])

# thread 2
with dist_autograd.context() as context_id:
  with ddp.no_sync():
    for _ in range(5):
      loss = ddp(input).sum()
      dist_autograd.backward(context_id, [loss])
  loss = ddp(input).sum()
  dist_autograd.backward(context_id, [loss])

input = torch.rand(10, requires_grad=True)
common = model1(input)

# thread1
with dist_autograd.context() as context_id:
    loss = model2(common).sum()
    dist_autograd.backward(context_id, [loss])

# thread2
with dist_autograd.context() as context_id:
    loss = model2(common).sum()
    dist_autograd.backward(context_id, [loss])

# thread3
loss = model3(common).sum()
loss.backward()