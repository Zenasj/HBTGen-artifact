import torch
import torch.nn as nn
import torch.jit


class Dense_Group(torch.jit.ScriptModule):
    __constants__ = ['groups']
    def __init__(self, in_feat, out_feat, groups=16):
        super().__init__()
        self.groups = groups

        in_feat_g = in_feat // groups
        out_feat_g = out_feat // groups

        assert in_feat_g * groups == in_feat, 'Found in_feat_g * groups != in_feat'
        assert out_feat_g * groups == out_feat, 'Found out_feat_g * groups != out_feat'

        dense_group = []
        for i in range(groups):
            den = nn.Linear(in_feat_g, out_feat_g, bias=True)
            dense_group.append(den)

        self.dense_group = nn.ModuleList(dense_group)

    # @torch.jit.script_method
    def forward(self, inputs):
        inputs_groups = torch.chunk(inputs, self.groups, 1)
        outputs_groups = []
        for i, m in enumerate(self.dense_group):
            outputs_groups.append(m(inputs_groups[i]))
        outputs = torch.cat(outputs_groups, 1)
        return outputs


class GenNet(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.dense1 = Dense_Group(z_dim, 200, 4)

    def forward(self, z_noise):
        y = self.dense1(z_noise)
        return y


dg = GenNet(128).cuda()
a = torch.rand(5, 128).cuda()
dgt = torch.jit.trace(dg, a)

import torch
import torch.nn as nn
import torch.jit


class Dense_Group(torch.jit.ScriptModule):
    __constants__ = ['groups']
    def __init__(self, in_feat, out_feat, groups=16):
        super(Dense_Group, self).__init__()
        self.groups = groups

        in_feat_g = in_feat // groups
        out_feat_g = out_feat // groups

        assert in_feat_g * groups == in_feat, 'Found in_feat_g * groups != in_feat'
        assert out_feat_g * groups == out_feat, 'Found out_feat_g * groups != out_feat'

        dense_group = []
        for i in range(groups):
            den = nn.Linear(in_feat_g, out_feat_g, bias=True)
            dense_group.append(den)

        self.dense_group = nn.ModuleList(dense_group)

    #torch.jit.script_method
    def forward(self, inputs):
        inputs_groups = torch.chunk(inputs, self.groups, 1)
        outputs_groups = []
        for i, m in enumerate(self.dense_group):
            outputs_groups.append(m(inputs_groups[i]))
        outputs = torch.cat(outputs_groups, 1)
        return outputs


class GenNet(nn.Module):
    def __init__(self, z_dim=128):
        super(GenNet, self).__init__()
        self.dense1 = Dense_Group(z_dim, 200, 4)

    def forward(self, z_noise):
        y = self.dense1(z_noise)
        return y


dg = GenNet(128).cuda()
a = torch.rand(5, 128).cuda()
dgt = torch.jit.trace(dg, a)