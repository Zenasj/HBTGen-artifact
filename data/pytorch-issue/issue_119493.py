import torch.nn as nn

py
import torch 

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

model = MyModule().eval().cuda()
input = torch.randn((1, 3, 224, 224), dtype=torch.float32)
ep = torch.export.export(model, (input,))
gm = ep.module()
print(getattr(gm, "conv_weight")) # Fails in 2.3
print(getattr(gm, "conv_weight")) # Fails in 2.3 but passes in 2.2

# Graph in 2.3 nightly 
# graph():
    # %conv_weight : [num_users=1] = get_attr[target=conv.weight]
    # %conv_bias : [num_users=1] = get_attr[target=conv.bias]
    # %l_x_ : [num_users=1] = placeholder[target=l_x_]
    # %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%l_x_, %conv_weight, %conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
    # %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
    # return (relu,)

# Graph in 2.2
# graph():
#     %conv_weight : [num_users=1] = get_attr[target=conv_weight]
#     %conv_bias : [num_users=1] = get_attr[target=conv_bias]
#     %l_x_ : [num_users=1] = placeholder[target=l_x_]
#     %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%l_x_, %conv_weight, %conv_bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#     %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#     return (relu,)