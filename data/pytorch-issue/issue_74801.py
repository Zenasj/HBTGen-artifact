from pathlib import Path
from typing import List

import torch
print(torch.__version__)

x1 = ([torch.empty((1, 10, 60, 80)), torch.empty((1, 10, 30, 40)), torch.empty((1, 10, 15, 20))], [8, 16, 32])


def test1(model_outs: List[torch.Tensor], strides: List[int] = (8, 16, 32)) -> torch.Tensor:
    # model_outs: List[Tensor] with len(model_outs) == len(strides), where each tensor is
    # of shape [batch_size, (cxcywh + obj + n_cls), height, width]

    out2 = []
    for i, x in enumerate(model_outs):
        stride = strides[i]

        # x = x.clone()  # Does not help
        x[:, 2:4] = torch.exp(x[:, 2:4]) * stride  # This causes failure
        out2.append(x)

        # # Workaround
        # wh = torch.exp(x[:, 2:4]) * stride
        # out2.append(torch.cat((x[:, :2], wh, x[:, 4:]), dim=1))

    out = [x.reshape(x.shape[0], x.shape[1], -1) for x in out2]  # flatten to [batch_size, n_channels, height * width]
    out = torch.cat(out, dim=2).permute(0, 2, 1)  # reshape to [batch_size, height * width, n_channels]

    return out

test1 = torch.jit.script(test1)
test1(*x1)
print(type(test1))
torch.onnx.export(test1, x1, Path('sth.onnx'),
                  opset_version=15)