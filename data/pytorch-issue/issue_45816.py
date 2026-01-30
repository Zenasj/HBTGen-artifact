import torch.nn as nn

@torch.jit.script
def yolo_predict(pred, anchors, stride: int, scale_x_y: float):
    #transpose
    nA = anchors.size(0)
    nB, nC, nH, nW = pred.size()
    nattr = int(nC / nA)
    pred = pred.view(nB, nA, nattr, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

    #make grid
    grid_x = torch.arange(nW).view(1, 1, 1, nW)
    grid_y = torch.arange(nH).view(1, 1, nH, 1)
    anchor_w = anchors[:, 0].view(1, nA, 1, 1).float() / (nW * stride)
    anchor_h = anchors[:, 1].view(1, nA, 1, 1).float() / (nH * stride)

    #apply grid
    x = pred[..., 0]
    y = pred[..., 1]
    w = pred[..., 2]
    h = pred[..., 3]
    p = pred[..., 4:]
    bx = ((torch.sigmoid(x) - 0.5) * scale_x_y + 0.5 + grid_x) / nW
    by = ((torch.sigmoid(y) - 0.5) * scale_x_y + 0.5 + grid_y) / nH
    bw = torch.exp(w) * anchor_w
    bh = torch.exp(h) * anchor_h
    bp = torch.sigmoid(p)
    preds = torch.stack([bx, by, bw, bh], -1)
    preds = torch.cat([preds, bp], -1)
    preds = preds.view(nB, -1, nattr)

    return preds

import torch

@torch.jit.script
def yolo_predict(pred, anchors, stride: int, scale_x_y: float):
    #transpose
    nA = anchors.shape[0]
    nB, nC, nH, nW = pred.shape
    nattr = int(nC / nA)
    pred = pred.view(nB, nA, nattr, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

    #make grid
    grid_x = torch.arange(nW).view(1, 1, 1, nW)
    grid_y = torch.arange(nH).view(1, 1, nH, 1)
    anchors_scaled = anchors.clone().float()
    anchors_scaled[:,0] /= (nW * stride)
    anchors_scaled[:,1] /= (nH * stride)
    anchor_w = anchors_scaled[:, 0].clone().view(1, nA, 1, 1)
    anchor_h = anchors_scaled[:, 1].clone().view(1, nA, 1, 1)

    #apply grid
    x = ((torch.sigmoid(pred[..., 0]) - 0.5) * scale_x_y + 0.5 + grid_x) / nW
    y = ((torch.sigmoid(pred[..., 1]) - 0.5) * scale_x_y + 0.5 + grid_y) / nH
    w = torch.exp(pred[..., 2]) * anchor_w
    h = torch.exp(pred[..., 3]) * anchor_h
    conf = pred[..., 4]
    cls  = pred[..., 5:]
    p = torch.sigmoid(pred[..., 4:])
    preds = torch.stack([x, y, w, h], -1)
    preds = torch.cat([preds, p], -1)
    preds = preds.view(nB, -1, nattr)

    # for inference we only need preds, but for training we need x, y, w, h, conf, cls and anchors_scaled
    return preds, x, y, w, h, conf, cls, anchors_scaled

class model(torch.nn.Module):
    def __init__(self, inc, nclasses):
        super(model, self).__init__()
        self.stride     = 8
        self.scale_x_y  = 1
        self.anchors    = torch.tensor([(10, 13), (16, 30), (33, 23)], requires_grad=False)
        self.conv       = torch.nn.Conv2d(inc, 3*(nclasses+5), 1, 1, 0, bias=True)


    def forward(self, x):
        x = self.conv(x)
        ret = yolo_predict(x, self.anchors, self.stride, self.scale_x_y)
        return ret[0]

if __name__ == '__main__':
    n = model(3, 80).eval()
    x = torch.randn(1,3,32,32)

    #inference runs fine
    y = n(x)
    print(y.shape)

    #export does not
    torch.onnx.export(n, (x,), "yolo_layer.onnx",
                      export_params=True,
                      opset_version=12,
                      verbose=True,
                      input_names=['x'],
                      output_names=['y'],
                      dynamic_axes={'x': [0, 2, 3], 'y': [0, 1]})

torch.onnx.export(...)