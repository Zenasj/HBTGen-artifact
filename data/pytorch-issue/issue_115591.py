import torch.distributed.checkpoint as dist_cp
dist_cp.load(state_dict={}, storage_reader=dist_cp.FileSystemReader('/tmp/foo'), no_dist=True)