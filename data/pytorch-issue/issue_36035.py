import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


gpus = list(map(int, sys.argv[1].split(',')))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = nn.ParameterList()

        for i in range(4):
            self.alpha.append(nn.Parameter(1e-3*torch.randn(i+2, 5)))

        self.cnn = nn.Conv2d(1, 1, 1, 1, 1)


    def forward(self, x):
        print(self.alpha)
        print(self.cnn)
        return x


if __name__ == '__main__':
    net = Net().cuda()
    if len(gpus) > 1:
        net = nn.DataParallel(net, device_ids=gpus)

    net(torch.rand(4, 5))

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


gpus = list(map(int, sys.argv[1].split(',')))


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = nn.ParameterList()
        for i in range(2):
            self.alpha.append(nn.Parameter(1e-3*torch.randn(i+2, 5)))
        print('Init: ', [a for a in self.alpha])


    def forward(self, x, alphas):
        # print('Inputs: ', x.shape)
        if alphas is not None:
            alphas = [a.squeeze(0) for a in alphas]
            print('In forward pass: ', [a for a in alphas])
            print([a.shape for a in alphas])
        else:
            print('In forward pass: ', [a.shape for a in self.alpha])
        return x


if __name__ == '__main__':
    net = Net().cuda()
    if len(gpus) > 1:
        net = nn.DataParallel(net, device_ids=gpus)
        alphas = [a.unsqueeze(0).repeat(len(gpus),1,1) for a in net.module.alpha]
        print([a.shape for a in alphas])
    else:
        alphas = None

    net(torch.rand(4, 5), alphas)
    # print('Not in forward pass: ', [n for n, p in net.named_parameters()])

if isinstance(module_copies[0][i], (ParameterList, ParameterDict)):
            # if replica is ParameterList or ParameterDict we setup its _parameters as
            # ordered dict of tensors
            for j in range(num_replicas):
                replica = module_copies[j][i]
                parameters = OrderedDict()
                for key, param in module._parameters.items():
                    param_idx = param_indices[param]
                    param = param_copies[j][param_idx]
                    parameters[str(key)] = param
                replica._parameters = parameters