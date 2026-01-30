import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    # normalization along channels
    def forward(self, x):
        # batch_size, channel, hight, width = x.size()
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()  # <---- here
        return x