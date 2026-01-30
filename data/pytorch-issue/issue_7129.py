import torch
import torch.nn as nn

class RemapByInds(nn.Module):
    def __init__(self, null_val=0):
        super(RemapByInds, self).__init__()
        self.null_val = null_val

    def forward(self, x, inds):
        time_steps_x = torch.ones_like(x[:, 0, :]).cumsum(dim=1) - 1
        batch_num_x = torch.ones_like(x[:, 0, :]).cumsum(dim=0) - 1
        out = torch.zeros_like(x) + self.null_val
        out_inds = torch.cat((time_steps_x[inds[:, 0], inds[:, 1]].view(-1, 1),
                              batch_num_x[inds[:, 0], inds[:, 1]].view(-1, 1)), dim=1).type(torch.long)
        out[out_inds[:, 0], :, out_inds[:, 1]] = x[inds[:, 0], :, inds[:, 1]]
        return out