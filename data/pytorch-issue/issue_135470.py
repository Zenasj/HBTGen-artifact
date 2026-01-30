import torch
import torch.nn as nn

from torch.export._trace import _export
exported_program: torch.export.ExportedProgram = _export(
    Mod(), args=example_args, strict=False, pre_dispatch=False
)
print(exported_program)
"""
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, sample_image: "f32[1, 32, 32]"):
             # File: /data/users/angelayi/pytorch2/moo.py:77 in forward, code: return (torch.fft.fft2(input = sample_image),)
            _to_copy: "c64[1, 32, 32]" = torch.ops.aten._to_copy.default(sample_image, dtype = torch.complex64);  sample_image = None
            _fft_c2c: "c64[1, 32, 32]" = torch.ops.aten._fft_c2c.default(_to_copy, [1, 2], 0, True);  _to_copy = None
            return (_fft_c2c,)
"""