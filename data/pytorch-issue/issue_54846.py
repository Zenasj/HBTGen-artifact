import torch.nn.functional as F

pool_outs = [F.max_pool1d(out, out.shape[2]).squeeze(2) for out in conv_outs]