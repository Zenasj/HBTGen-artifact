from __future__ import absolute_import

import torch
import torch.nn as nn
import math
from torch.autograd import Function, Variable, grad
import numpy as np
from itertools import chain


__all__ = ['gunn']


class Gunn_function(Function):
    @staticmethod
    def forward(ctx, x, list_of_modules, *parameters_of_list_of_modules):
        # ctx.mark_dirty(x)  # 0.3
        # x = Variable(x, volatile=True)  # 0.3

        ctx.gather, ctx.updater, ctx.scatter = list_of_modules

        var_temp = ctx.updater(ctx.gather(x))
        var_dx = ctx.scatter(var_temp)

        x.data.add_(var_dx.data)

        ctx.x = x.data
        ctx.temp = var_temp.data

        # x = x.data  # 0.3
        return x

    @staticmethod
    def backward(ctx, gradient):
        with torch.enable_grad():
            var_temp = Variable(ctx.temp, requires_grad=True)
            var_dx = ctx.scatter(var_temp)

            ctx.x.add_(-var_dx.data)  # change x back to input

            var_x = Variable(ctx.x, requires_grad=True)
            var_temp2 = ctx.updater(ctx.gather(var_x))

        parameters_tuple1 = tuple(filter(lambda x: x.requires_grad, ctx.scatter.parameters()))
        parameters_tuple2 = tuple(filter(lambda x: x.requires_grad, chain(ctx.gather.parameters(), ctx.updater.parameters())))
        temp_grad, *parameters_grads1 = torch.autograd.grad(var_dx, (var_temp,) + parameters_tuple1, gradient)
        x_grad, *parameters_grads2 = torch.autograd.grad(var_temp2, (var_x,) + parameters_tuple2, temp_grad)

        return (x_grad + gradient, None, ) + tuple(parameters_grads2 + parameters_grads1)


class Update(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(Update, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * K, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * K)
        self.conv2 = nn.Conv2d(out_channels * K, out_channels * K, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * K)
        self.conv3 = nn.Conv2d(out_channels * K, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv4.forward(x)
        y = self.bn4.forward(y)

        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = self.relu.forward(x)

        x = self.conv3.forward(x)
        x = self.bn3.forward(x)
        return x + y


def get_tensor(L, P, p, type='gunn'):
    if type == 'identity':
        return torch.eye(L * P)
    elif type == 'gunn':
        return torch.cat([torch.zeros(L * p, L), torch.eye(L * (P - p), L)], dim=0)
    raise NotImplementedError


class Gunn_layer(nn.Module):
    def __init__(self, N, P, K):
        super(Gunn_layer, self).__init__()
        self.P = P
        L = N // P

        for p in range(P):
            # gather = nn.Conv2d(N, L, kernel_size=1, bias=False)
            # gather.weight.data = get_tensor(L, P, p, 'identity').t().unsqueeze(2).unsqueeze(3)
            # gather.weight.requires_grad = False
            # gather.inited = True
            gather = nn.Sequential()
            updater = Update(N, L, K)

            scatter = nn.Conv2d(L, N, kernel_size=1, bias=False)
            scatter.weight.data = get_tensor(L, P, p, 'gunn').unsqueeze(2).unsqueeze(3)
            scatter.weight.requires_grad = False
            scatter.inited = True

            self.add_module('gather' + str(p), gather)
            self.add_module('updater' + str(p), updater)
            self.add_module('scatter' + str(p), scatter)

    def forward(self, x):
        for p in range(self.P):
            parameters = list(filter(lambda param: param.requires_grad, chain(self._modules['gather' + str(p)].parameters(), self._modules['updater' + str(p)].parameters(), self._modules['scatter' + str(p)].parameters())))
            modules = (self._modules['gather' + str(p)], self._modules['updater' + str(p)], self._modules['scatter' + str(p)])
            x = Gunn_function.apply(x, modules, *parameters)
        return x


class Gunn(nn.Module):
    def __init__(self, num_classes=10):
        super(Gunn, self).__init__()
        self.num_classes = num_classes
        N1 = 240
        N2 = 300
        N3 = 360

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.trans0_conv = nn.Conv2d(64, N1, kernel_size=1, bias=False)
        self.trans0_bn = nn.BatchNorm2d(N1)

        self.layer1 = Gunn_layer(N1, 20, 2)

        self.trans1_conv = nn.Conv2d(N1, N2, kernel_size=1, bias=False)
        self.trans1_bn = nn.BatchNorm2d(N2)

        self.layer2 = Gunn_layer(N2, 25, 2)

        self.trans2_conv = nn.Conv2d(N2, N3, kernel_size=1, bias=False)
        self.trans2_bn = nn.BatchNorm2d(N3)

        self.layer3 = Gunn_layer(N3, 30, 2)

        self.trans3_conv = nn.Conv2d(N3, N3, kernel_size=1, bias=False)
        self.trans3_bn = nn.BatchNorm2d(N3)

        self.avgpool = nn.AvgPool2d(2, 2)
        self.GAP = nn.AvgPool2d(8)
        self.fc = nn.Linear(N3, num_classes)

        for m in self.modules():
            if hasattr(m, 'inited'):
                continue
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.trans0_conv(x)
        x = self.trans0_bn(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32

        x = self.trans1_conv(x)
        x = self.trans1_bn(x)
        x = self.relu(x)
        x = self.avgpool(x)  # 16x16

        x = self.layer2(x)

        x = self.trans2_conv(x)
        x = self.trans2_bn(x)
        x = self.relu(x)
        x = self.avgpool(x)  # 8x8

        x = self.layer3(x)  # 8x8

        x = self.trans3_conv(x)
        x = self.trans3_bn(x)
        x = self.relu(x)
        x = self.GAP(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)

        return output


def gunn(**kwargs):
    return Gunn(**kwargs)