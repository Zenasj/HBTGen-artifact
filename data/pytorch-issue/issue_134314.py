# test.py
import torch
import os
rank = int(os.getenv("LOCAL_RANK"))
torch.distributed.init_process_group('nccl', device_id=torch.device(rank))
g = torch.distributed.new_group(ranks=[0])
torch.distributed.barrier()
print("Done!")