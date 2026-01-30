import torch

torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(hours=2.0).total_seconds())

torch.distributed.init_process_group('nccl', timeout=datetime.timedelta(hours=2.0))