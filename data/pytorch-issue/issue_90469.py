import torch
import torch.nn as nn
import torch.nn.functional as F

def forward(self, x):
    self_linear = self.self_linear(x);  x = None
    tensor = torch.tensor(2)
    mul = self_linear * tensor;  self_linear = tensor = None
    mul_1 = mul * mul;  mul = mul = None
    return mul_1

def forward(self, x):
    self_linear = self.self_linear(x);  x = None
    tensor = torch.tensor(2)
    mul = self_linear * tensor;  self_linear = tensor = None
    mul_1 = mul * add;  mul = mul = None
    return mul_1

class GraphModule(torch.nn.Module):
    def forward(self, pred, x):
        arg0: "b8[]"; arg1: "f32[3, 3]"; 
    
        arg0, arg1, = fx_pytree.tree_flatten_spec(([pred, x], {}), self._in_spec)
        l_pred_ = arg0
        l_x_ = arg1
        
        # File: /home/yidi/local/pytorch/test/dynamo/test_export.py:1535, code: y = x * 2
        y = l_x_ * 2
        
        # File: /home/yidi/local/pytorch/torch/nn/modules/linear.py:116, code: return F.linear(input, self.weight, self.bias)
        l__self___linear_weight = self.L__self___linear_weight
        l__self___linear_bias = self.L__self___linear_bias
        
        # File: /home/yidi/local/pytorch/torch/_higher_order_ops/cond.py:116, code: return cond_op(pred, true_fn, false_fn, operands)
        cond_true_0 = self.cond_true_0
        cond_false_0 = self.cond_false_0
        cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, [l__self___linear_bias, l__self___linear_weight, l_x_, y]);  l_pred_ = cond_true_0 = cond_false_0 = l__self___linear_bias = l__self___linear_weight = l_x_ = y = None
        getitem = cond[0];  cond = None
        return pytree.tree_unflatten([getitem], self._out_spec)
        
    class GraphModule(torch.nn.Module):
        def forward(self, l__self___linear_bias, l__self___linear_weight, l_x_, y_true_branch):
            l__self___linear_bias_1 = l__self___linear_bias
            l__self___linear_weight_1 = l__self___linear_weight
            l_x__1 = l_x_
            
            # File: /home/yidi/local/pytorch/torch/nn/modules/linear.py:116, code: return F.linear(input, self.weight, self.bias)
            linear = torch._C._nn.linear(l_x__1, l__self___linear_weight_1, l__self___linear_bias_1);  l_x__1 = l__self___linear_weight_1 = l__self___linear_bias_1 = None
            
            # File: /home/yidi/local/pytorch/test/dynamo/test_export.py:1537, code: return self.linear(val) * torch.tensor(2) * y
            tensor = torch.tensor(2)
            mul = linear * tensor;  linear = tensor = None
            mul_1 = mul * y_true_branch;  mul = y_true_branch = None
            return (mul_1,)
            
    class GraphModule(torch.nn.Module):
        def forward(self, l__self___linear_bias_1, l__self___linear_weight_1, l_x_, y_true_branch):
            l__self___linear_bias_2 = l__self___linear_bias_1
            l__self___linear_weight_2 = l__self___linear_weight_1
            l_x__1 = l_x_
            
            # File: /home/yidi/local/pytorch/torch/nn/modules/linear.py:116, code: return F.linear(input, self.weight, self.bias)
            linear = torch._C._nn.linear(l_x__1, l__self___linear_weight_2, l__self___linear_bias_2);  l_x__1 = l__self___linear_weight_2 = l__self___linear_bias_2 = None
            
            # File: /home/yidi/local/pytorch/test/dynamo/test_export.py:1540, code: return self.linear(val) * torch.tensor(-1)
            tensor = torch.tensor(-1)
            mul = linear * tensor;  linear = tensor = None
            return (mul,)