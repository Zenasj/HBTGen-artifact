# torch.rand(B, 3, H, W, dtype=torch.float32)

import torch
import torch.nn as nn

@torch.jit.script
def yolo_predict(pred, anchors, stride: int, scale_x_y: float):
    nA = anchors.size(0)
    nB = pred.size(0)
    nC = pred.size(1)
    nH = pred.size(2)
    nW = pred.size(3)
    nattr = nC // nA
    pred = pred.view(nB, nA, nattr, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

    grid_x = torch.arange(nW, device=pred.device).view(1, 1, 1, nW)
    grid_y = torch.arange(nH, device=pred.device).view(1, 1, nH, 1)
    anchor_w = (anchors[:, 0].view(1, nA, 1, 1).float() / (nW * stride))
    anchor_h = (anchors[:, 1].view(1, nA, 1, 1).float() / (nH * stride))

    x = pred[..., 0]
    y = pred[..., 1]
    w = pred[..., 2]
    h = pred[..., 3]
    p = pred[..., 4:]

    bx = ((torch.sigmoid(x) - 0.5) * scale_x_y + 0.5 + grid_x) / nW
    by = ((torch.sigmoid(y) - 0.5) * scale_x_y + 0.5 + grid_y) / nH
    bw = torch.exp(w) * anchor_w
    bh = torch.exp(h) * anchor_h
    p = torch.sigmoid(p)

    preds = torch.stack([bx, by, bw, bh], dim=-1)
    preds = torch.cat([preds, p], dim=-1)
    preds = preds.view(nB, -1, nattr)
    return preds, x, y, w, h, p

class MyModel(nn.Module):
    def __init__(self, inc=3, nclasses=80):
        super(MyModel, self).__init__()
        self.stride = 8
        self.scale_x_y = 1.0
        self.register_buffer('anchors', torch.tensor(
            [[10, 13], [16, 30], [33, 23]], dtype=torch.float32))
        self.conv = nn.Conv2d(inc, 3 * (nclasses + 5), 1, 1, 0, bias=True)

    def forward(self, x):
        x = self.conv(x)
        return yolo_predict(x, self.anchors, self.stride, self.scale_x_y)[0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 32, 32, dtype=torch.float32)

