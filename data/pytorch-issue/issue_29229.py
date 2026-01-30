# conda install pytorch torchvision -c pytorch-nightly
# conda install -c conda-forge onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        # m = nn.UpsamplingBilinear2d(scale_factor=2)  # fails
        # m = nn.UpsamplingNearest2d(scale_factor=2)  # fails
        x = F.interpolate(x, scale_factor=2)  # fails
        return x


print('torch version %s' % torch.__version__)
print('onnx version %s' % onnx.__version__)
img = torch.randn(1, 3, 256, 256)
tml = TestModel()
print(img.shape, tml(img).shape)
torch.onnx.export(tml, img, 'model.onnx', verbose=False, opset_version=11)

oml = onnx.load('model.onnx')  # onnx model
onnx.checker.check_model(oml)