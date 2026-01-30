import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


if __name__ == '__main__':
    device = torch.device('cuda')

    layernorm = LayerNorm2d(3).to(device)
    conv = nn.Conv2d(3, 3, kernel_size=1).to(device)

    x = torch.ones(4, 3, 224, 224, device=device, requires_grad=True)
    x = layernorm(x)
    x = checkpoint(conv, x)

    loss = torch.mean(x)
    loss.backward()