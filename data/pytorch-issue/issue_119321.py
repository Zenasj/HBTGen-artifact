import torch
import torch.nn as nn

def test_kjt_export2(self):

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.split([1, 2, 1])

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        m = KJTInputExportWrapper(M(), kjt.keys())
        pt2_ir = torch.export.export(m, (kjt.values(), kjt.lengths()), {}, strict=False)
        print(pt2_ir)
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [5, 0, 0, 0])
        actual_output = pt2_ir.module()(kjt.values(), kjt.lengths())
        exp_output = M()(kjt)