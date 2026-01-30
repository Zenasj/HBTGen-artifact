with ddp.no_sync():
  for input in inputs:
    ddp(input).backward()

ddp(one_more_input).backward() 
optimizer.step()