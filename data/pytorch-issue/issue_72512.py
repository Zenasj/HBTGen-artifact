import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


if __name__ == '__main__':
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    x = torch.ones(4, 3, 224, 224, device=device, requires_grad=True)

    conv1 = nn.Conv2d(3, 96, kernel_size=1).to(device)
    ln = LayerNorm2d(96).to(device)
    conv2 = nn.Conv2d(96, 96, kernel_size=3, groups=3).to(device)

    x = ln(conv1(x))

    o1 = conv2(x)
    o2 = conv2(x.contiguous())

    diff = o1 - o2
    print(torch.min(diff), torch.max(diff))