# torch.rand(B, 3, 112, 112, dtype=torch.float32)  # Input shape for ResNet18 with SEBlock
import torch
import torch.nn as nn
from torch.nn import functional as F

class PReLU_Quantized(nn.Module):
    def __init__(self, prelu_object):
        super().__init__()
        self.prelu_weight = prelu_object.weight
        self.weight = self.prelu_weight
        self.quantized_op = nn.quantized.FloatFunctional()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, inputs):
        self.weight = self.quant(self.weight)
        weight_min_res = self.quantized_op.mul(-self.weight, torch.relu(-inputs))
        inputs = self.quantized_op.add(torch.relu(inputs), weight_min_res)
        inputs = self.dequant(inputs)
        self.weight = self.dequant(self.weight)
        return inputs

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mult_xy = nn.quantized.FloatFunctional()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.fc1 = self.fc[0]
        self.prelu = self.fc[1]
        self.fc2 = self.fc[2]
        self.sigmoid = self.fc[3]
        self.prelu_q = PReLU_Quantized(self.prelu)  # Quantized PReLU for SEBlock

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.prelu_q(y)  # Problematic PReLU quantization step
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        out = self.mult_xy.mul(x, y)
        return out

class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.prelu_q = PReLU_Quantized(self.prelu)
        self.conv2 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
        self.add_residual_relu = nn.quantized.FloatFunctional()

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu_q(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.add_residual_relu.add_relu(out, residual)
        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)

class ResNet(nn.Module):
    def __init__(self, block, layers, use_se=True):
        super().__init__()
        self.inplanes = 64
        self.use_se = use_se
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.prelu_q = PReLU_Quantized(self.prelu)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)  # Matches 7x7 spatial dimension
        self.bn3 = nn.BatchNorm1d(512)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu_q(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1']], inplace=True)
        for m in self.modules():
            if isinstance(m, IRBlock):
                m.fuse_model()

class MyModel(nn.Module):  # Single fused model as per issue's context
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = ResNet(IRBlock, [2, 2, 2, 2], use_se=True)  # ResNet18 with SEBlocks

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 112, 112, dtype=torch.float32)  # 112x112 input for 7x7 FC spatial dim

