import torch
import torch.nn.functional as F

mask_sfmx = - 1e+10 * torch.logical_not(attn_mask)[:, :, None]
weight = F.softmax(self.attn(input_embs) + mask_sfmx, dim=1)
mean = torch.sum(input_embs*weight, dim=1)