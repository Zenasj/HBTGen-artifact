# torch.rand(2, 64, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, cardinality, bottleneck_width, stride):
        super().__init__()
        self.bwidth = bottleneck_width
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * bottleneck_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * bottleneck_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * bottleneck_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * bottleneck_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * bottleneck_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ModuleListBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, cardinality, bottleneck_width, stride):
        super().__init__()
        self.bwidth = bottleneck_width
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.ModuleList([
            nn.Conv2d(bottleneck_width, bottleneck_width, kernel_size=3, stride=stride, padding=1, bias=False)
            for _ in range(cardinality)
        ])
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion * bottleneck_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * bottleneck_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * bottleneck_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * bottleneck_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * bottleneck_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        split_out = []
        for n, m in enumerate(self.conv2):
            start = n * self.bwidth
            end = (n + 1) * self.bwidth
            split_out.append(m(out[:, start:end, :, :]))
        out = torch.cat(split_out, 1)
        out = F.relu(self.bn2(out))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MyModel(nn.Module):
    def __init__(self, in_planes, cardinality, bottleneck_width, stride):
        super().__init__()
        self.block1 = GroupedBlock(in_planes, cardinality, bottleneck_width, stride)
        self.block2 = ModuleListBlock(in_planes, cardinality, bottleneck_width, stride)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(x)
        return torch.max(torch.abs(out1 - out2))  # Return maximum absolute difference


def my_model_function():
    return MyModel(in_planes=64, cardinality=8, bottleneck_width=64, stride=1)


def GetInput():
    return torch.rand(2, 64, 32, 32, dtype=torch.float32)

