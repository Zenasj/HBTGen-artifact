import torch

ag = torch.ops.aten.all_gather_into_tensor(c, tag, ranks, group_size)
# must call wait_tensor on ag before peeking at ag's data, it's not gauranteed valid yet.  shape/size/dtype are OK to peek at.  wait_tensor ensures underneath cuda stream used for comms is synced.
ag = torch.ops.aten.wait_tensor(ag)