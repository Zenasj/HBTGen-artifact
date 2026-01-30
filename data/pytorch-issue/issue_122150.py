import torch.nn as nn

import torch


class Module(torch.nn.Module):
    def forward(self, x):
        x, mx = torch.ops.aten.native_dropout(x, 0.1, True)
        y, my = torch.ops.aten.native_dropout(x, 0.1, True)
        return mx, my


eo = torch.onnx.dynamo_export(Module(), torch.rand(1, 2, 3))
import onnx

onnx.printer.to_text(eo.model_proto)

"""
...
main_graph (float[1,2,3] l_x_) => (bool[1,2,3] native_dropout_1, bool[1,2,3] native_dropout_1_1) 
   <float[1,2,3] native_dropout, float[1,2,3] native_dropout_1, float[1,2,3] l_x_, bool[1,2,3] native_dropout_1, bool[1,2,3] native_dropout_1_1>
{
   native_dropout, native_dropout_1.1 = pkg.onnxscript.torch_lib.aten_native_dropout <p: float = 0.1, train: int = 1> (l_x_)
   native_dropout_1, native_dropout_1_1 = pkg.onnxscript.torch_lib.aten_native_dropout <p: float = 0.1, train: int = 1> (native_dropout)
}
...
Graph first output:
name: "native_dropout_1"
type {
  tensor_type {
    elem_type: 9
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 2
      }
      dim {
        dim_value: 3
      }
    }
  }
}

First node outputs
['native_dropout', 'native_dropout_1.1']
Second node outputs
['native_dropout_1', 'native_dropout_1_1']
"""