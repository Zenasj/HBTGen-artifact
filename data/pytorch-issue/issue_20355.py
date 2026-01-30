import torch

for input in inputs:
  output = model(input)
  loss_func(output, label).backward()
  all_reduce(model.parameters(), ..., async_op=True) # only use all reduce to achieve model averaging.

for i in range(n_batches):
  optim.zero_grad()
  input = prepare_batch(i) #heaviest operation is torch.index_select, which is multithreaded
  for j in range(len(input)):  #whole input doesnt fit in gpu, and loss function is additive
    output = model(input[j])
    loss_func(output, label).backward()
  average_gradients(optim, async_op=False)
  optim.step()