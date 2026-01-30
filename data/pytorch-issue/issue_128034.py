import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphModule(torch.nn.Module):
  def forward(self, L_x_: "f32[1024, 1024]"):
      l_x_ = L_x_
      
      # File: /data/users/lsakka/pytorch/pytorch/test/dynamo/test_structured_trace.py:284 in forward, code: return self.layers(x)
      l__self___layers_0: "f32[1024, 1024]" = self.L__self___layers_0(l_x_);  l_x_ = None
      l__self___layers_1: "f32[1024, 1024]" = self.L__self___layers_1(l__self___layers_0);  l__self___layers_0 = None
      return (l__self___layers_1,)

class GraphModule(torch.nn.Module):
    def forward(self, L_self_layers_0_weight: "f32[1024, 1024]", L_self_layers_0_bias: "f32[1024]", L_x_: "f32[1024, 1024]", L_self_layers_1_weight: "f32[1024, 1024]", L_self_layers_1_bias: "f32[1024]"):
        l_self_layers_0_weight = L_self_layers_0_weight
        l_self_layers_0_bias = L_self_layers_0_bias
        l_x_ = L_x_
        l_self_layers_1_weight = L_self_layers_1_weight
        l_self_layers_1_bias = L_self_layers_1_bias
        
        # File: /data/users/lsakka/pytorch/pytorch/torch/nn/modules/linear.py:116 in forward, code: return F.linear(input, self.weight, self.bias)
        input_1: "f32[1024, 1024]" = torch._C._nn.linear(l_x_, l_self_layers_0_weight, l_self_layers_0_bias);  l_x_ = l_self_layers_0_weight = l_self_layers_0_bias = None
        input_2: "f32[1024, 1024]" = torch._C._nn.linear(input_1, l_self_layers_1_weight, l_self_layers_1_bias);  input_1 = l_self_layers_1_weight = l_self_layers_1_bias = None
        return (input_2,)