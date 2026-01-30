backend = 'nccl'
init_process_group(backend=backend, timeout=timedelta(seconds=60))