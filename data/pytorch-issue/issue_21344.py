m = SomeModel()
dist.init_process_group()
ddp = DistributedDataParallel(m)
# got error
dist.destroy_process_group()
dist.init_process_group()
del ddp
ddp = DistributedDataParallel(m)