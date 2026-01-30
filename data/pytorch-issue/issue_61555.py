ddp_model = DistributedDataParallel(...)
zero_optim = ZeroRedundancyOptimizer(ddp_model.parameters(), ...)
with _Join([ddp_model, zero_optim]):
    ...

with _Join([ddp_model, zero_optim], divide_by_initial_world_size=False):
    ...