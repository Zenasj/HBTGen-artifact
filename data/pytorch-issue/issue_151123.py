import torch

def forward(self):
        arg9_1: "f32[10, 2139]"

        _tensor_constant0: "f32[1]" = self._tensor_constant0 # this should be int64, conflicted with the original _tensor_constant0, had a clone on this constant before lifting


        index: "f32[10, 925]" = torch.ops.aten.index.Tensor(arg9_1, [None, _tensor_constant0]);  _tensor_constant0 = None