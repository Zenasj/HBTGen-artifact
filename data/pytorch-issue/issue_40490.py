import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.quantization.utils import _replace_relu, quantize_model

class QuantBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(QuantBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3,stride=4, padding=1, groups=inplanes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, 1,stride, 0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.reduce = nn.Conv2d(inplanes, planes//4, 1,stride, 0)
        self.expand = nn.Conv2d(inplanes//4, planes, 1,stride, 0)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=4.0)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.reduce(out)
        out = self.expand(out)
        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2', 'relu']], inplace=True)


class QuantizableResNet(nn.Module):

    def __init__(self, block, layers, **kwargs):
        super(QuantizableResNet, self).__init__()
        self.inplanes = 256
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3,self.inplanes, 3, 16, 1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.inplanes, 100)
        self.dequant = torch.quantization.DeQuantStub()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantBasicBlock:
                m.fuse_model()


def _resnet(backend):
    model = QuantizableResNet(QuantBasicBlock, [50])
    _replace_relu(model)
    quantize_model(model, backend)

    return model


if __name__ == "__main__":
    backend = 'qnnpack'
    qmodel = _resnet(backend)
    qmodel.eval()
    qts = torch.jit.script(qmodel)
    jit_path = "bug_{}.pt".format(backend)
    torch.jit.save(qts, jit_path)

run #1: 0.8429839611053467
run #2: 0.21468663215637207
run #3: 0.2053544521331787