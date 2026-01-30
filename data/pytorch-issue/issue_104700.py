import torch.nn as nn

from torch_scatter import scatter_max
import torch

class MyModel(torch.nn.Module):
    def forward(self, src: torch.Tensor, idx: torch.Tensor):
        return scatter_max(src, idx)

m = MyModel().eval()
src = torch.ones([3,10], dtype=torch.float32)
idx = torch.randint(0, 4, [3, 10], dtype=torch.long)

def sym_scatter_max(g, src, index, dim, out, dim_size):
    return g.op(
                "torch_scatter::scatter_max",
                src,
                index,
                dim_size_i=-1,
                outputs=2
            )
torch.onnx.register_custom_op_symbolic('torch_scatter::scatter_max', sym_scatter_max, 1)

with torch.no_grad():
    torch.onnx.export(m, (src, idx), 'mymodel.onnx', verbose=False, opset_version=13, custom_opsets={'torch_scatter':1}, do_constant_folding=True)