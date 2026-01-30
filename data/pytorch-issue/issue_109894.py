import torch
import torch.nn as nn

class GraphModule(torch.nn.Module):
    def forward(self, arg23_1: bf16[5, 1, 28, 28]):
        # File: /home/binbao/pytorch/torch/nn/modules/container.py:215, code: input = module(input)
        _0_weight_1: bf16[64, 1, 3, 3] = getattr(self, "0_weight")
        _0_bias_1: bf16[64] = getattr(self, "0_bias")
        _1_weight_1: bf16[64] = getattr(self, "1_weight")
        _1_bias_1: bf16[64] = getattr(self, "1_bias")
        _4_weight_1: bf16[64, 64, 3, 3] = getattr(self, "4_weight")
        _4_bias_1: bf16[64] = getattr(self, "4_bias")
        _5_weight_1: bf16[64] = getattr(self, "5_weight")
        _5_bias_1: bf16[64] = getattr(self, "5_bias")
        _8_weight_1: bf16[64, 64, 3, 3] = getattr(self, "8_weight")
        _8_bias_1: bf16[64] = getattr(self, "8_bias")
        _9_weight_1: bf16[64] = getattr(self, "9_weight")
        _9_bias_1: bf16[64] = getattr(self, "9_bias")
        _13_weight_1: bf16[5, 64] = getattr(self, "13_weight")
        _13_bias_1: bf16[5] = getattr(self, "13_bias")
        _1_running_mean_1: bf16[64] = getattr(self, "1_running_mean")
        _1_running_var_1: bf16[64] = getattr(self, "1_running_var")
        _5_running_mean_1: bf16[64] = getattr(self, "5_running_mean")
        _5_running_var_1: bf16[64] = getattr(self, "5_running_var")
        _9_running_mean_1: bf16[64] = getattr(self, "9_running_mean")
        _9_running_var_1: bf16[64] = getattr(self, "9_running_var")
        arg23_2 = arg23_1
        convolution: bf16[5, 64, 26, 26] = torch.ops.aten.convolution.default(arg23_2, _0_weight_1, _0_bias_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg23_2 = _0_weight_1 = _0_bias_1 = None