import torch
import torch.nn as nn
import torch.nn.functional as F

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
    def forward(self, x):

        #return F.avg_pool2d(x, (480, 640))  # this is ok

        return F.avg_pool2d(x, x.size()[2:])
                # RuntimeError: ONNX symbolic expected a constant value in the trace

        #return F.adaptive_avg_pool2d(x, (1, 1))  # this is ok

net = mynet()
x = torch.randn(1, 3, 480, 640)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
x = x.to(device)

out = net(x)
print('out.size ', out.size()) #(1, 3, 1, 1)

torch.onnx.export(net, x, "test.onnx", verbose=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
    def forward(self, x):
        n, c, h, w = x.size()
        x = F.adaptive_avg_pool2d(x, (1, 1))

        #return F.interpolate(x, (h, w), mode='bilinear', align_corners=False)
                                # RuntimeError: ONNX symbolic expected a constant value in the trace

        return F.interpolate(x, (480, 640), mode='bilinear', align_corners=False) # this is ok

        #return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
                                # RuntimeError: ONNX symbolic expected a constant value in the trace

        #return F.interpolate(x, (480, 640), mode='bilinear', align_corners=True)
                                # UserWarning: ONNX export failed on upsample_bilinear2d because align_corners == True not supported
                                # RuntimeError: ONNX export failed: Couldn't export operator aten::upsample_bilinear2d


net = mynet()
x = torch.randn(1, 3, 480, 640)

device = torch.device("cuda:0")
net = net.to(device)
x = x.to(device)

out = net(x)
print('out.size ', out.size()) #(1, 3, 480, 640)

torch.onnx.export(net, x, "test.onnx", verbose=True)