import torch

view

view

[64//s3, s3, 4, 49, 49]

Sym(s3)

sym_size_16: Sym(s3) = torch.ops.aten.sym_size(arg34_1, 0)
floordiv_3: Sym(64//s3) = sym_size_13 // sym_size_16
view_33: f32[64//s3, 64//(64//s3), 4, 49, 49] = torch.ops.aten.view.default(add_12, [floordiv_3, sym_size_16, 4, sym_size_14, sym_size_14]);  add_12 = floordiv_3 = sym_size_16 = None

Sym(64//s3)

1