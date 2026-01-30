import torch

torch.distributed.init_process_group('gloo')
torch.distributed.destroy_process_group()

raise ZeroDivisionError