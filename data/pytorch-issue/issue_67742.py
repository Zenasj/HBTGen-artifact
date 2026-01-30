dist.init_process_group("nccl")

dist.init_process_group("nccl", timeout=timedelta(seconds=10))