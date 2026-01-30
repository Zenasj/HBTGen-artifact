import torch.nn as nn

import torch

torch.manual_seed(0)

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        t1 = torch.unbind(x)
        t2 = torch.stack(t1, dim=1)
        t3 = torch.tanh(t2)
        return t3

func = Model().to('cpu')

x = torch.randn(2, 3, 4)

with torch.no_grad():
    func1 = torch.compile(func)
    print(func1(x.clone()).shape)
    """
    File "torch/_inductor/fx_passes/split_cat.py", line 1085, in merge_unbind_stack
    UnbindCatRemover().remove_unbind(match.graph, unbind_node)
    File "torch/_inductor/fx_passes/split_cat.py", line 890, in remove_unbind
        super().simplify(graph, unbind_node, split_sections)
    File "torch/_inductor/fx_passes/split_cat.py", line 500, in simplify
        transform_params_list = self.get_transform_params(
    File "torch/_inductor/fx_passes/split_cat.py", line 932, in get_transform_params
        split_dim = unbind_node.kwargs["dim"]
    torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    KeyError: 'dim'
    """