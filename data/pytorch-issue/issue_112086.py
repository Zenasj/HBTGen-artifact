import torch

args = (torch.randn(768, 48, 128), torch.randn(1, 1, 128))


def func(x, y):
  return torch.ops.aten._weight_norm_interface(x, y, 2)

func(args[0], args[1])
exp = torch.export.export(func, args)
exp = exp.run_decompositions()
print(exp.graph_module.code)

def forward(self, arg0_1, arg1_1):
    _weight_norm_interface = torch.ops.aten._weight_norm_interface.default(arg0_1, arg1_1, 2);  arg0_1 = arg1_1 = None
    getitem = _weight_norm_interface[0]
    getitem_1 = _weight_norm_interface[1];  _weight_norm_interface = None
    return (getitem, getitem_1)