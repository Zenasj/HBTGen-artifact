import torch.nn as nn

py
import torch

torch.manual_seed(420)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v1.resize_(1, 1, 2)
        v2 = torch.nn.functional.linear(v1, self.linear.weight, self.linear.bias)
        return v2

func = Model().cuda()

x = torch.randn(1, 2, 2).cuda()


with torch.no_grad():
    func.train(False)
    jit_func = torch.compile(func)

    res1 = func(x) # without jit
    print(res1)

    res2 = jit_func(x)
    print(res2)

    torch.testing.assert_close(res1, res2, rtol=1e-3, atol=1e-3)

v1_updated = v1.as_strided((1, 1, 2), (2, 2, 1))
v2 = torch.nn.functional.linear(v1_updated, self.linear.weight, self.linear.bias)

def forward(self, arg0_1, arg1_1, arg2_1):
    permute = torch.ops.aten.permute.default(arg0_1, [0, 2, 1])
    resize = torch.ops.aten.resize.default(permute, [1, 1, 2])
    as_strided = torch.ops.aten.as_strided.default(permute, [1, 1, 2], [2, 2, 1]);  permute = None
    permute_1 = torch.ops.aten.permute.default(arg0_1, [0, 2, 1]);  arg0_1 = None
    permute_2 = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
    expand = torch.ops.aten.expand.default(permute_1, [1, 2, 2]);  permute_1 = None
    view = torch.ops.aten.view.default(expand, [1, 2, 2]);  expand = None
    expand_1 = torch.ops.aten.expand.default(permute_2, [1, 2, 2]);  permute_2 = None
    view_1 = torch.ops.aten.view.default(expand_1, [1, 2, 2]);  expand_1 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [1, 2, 2]);  bmm = None
    add = torch.ops.aten.add.Tensor(view_2, arg2_1);  view_2 = arg2_1 = None
    return (add,)