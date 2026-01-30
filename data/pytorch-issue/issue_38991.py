import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        # m = nn.UpsamplingBilinear2d(scale_factor=2)  # also occurs
        # m = nn.UpsamplingNearest2d(scale_factor=2)  # also occurs

    def forward(self, x):
        # return m(x)
        return F.interpolate(x, scale_factor=2)


print('torch version %s' % torch.__version__)
print('onnx version %s' % onnx.__version__)
img = torch.randn(1, 3, 256, 256)
tml = TestModel()
print(img.shape, tml(img).shape)
torch.onnx.export(tml, img, 'model.onnx', verbose=False, opset_version=11)

oml = onnx.load('model.onnx')  # onnx model
onnx.checker.check_model(oml)

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.m = nn.Upsample(scale_factor=2, recompute_scale_factor=False)

    def forward(self, x):
        return self.m(x)