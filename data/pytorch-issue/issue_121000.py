import torch.nn as nn

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention

class Module(torch.nn.Module):
    def forward(
        self, query: torch.Tensor, cache: torch.Tensor, start_pos: torch.Tensor
    ) -> torch.Tensor:
        # x.sizes(): 1, 128, 16, 128
        sp = start_pos.item()
        torch._constrain_as_size(sp, min=0, max=126)
        key = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
        value = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/attention.cpp#L732
        return scaled_dot_product_attention(query, key, value)


cache = torch.randn(1, 128, 16, 128, dtype=torch.float16)
query = torch.randn(1, 1, 16, 128, dtype=torch.float16)
start_pos = torch.tensor([0])
with sdpa_kernel(SDPBackend.MATH), torch.no_grad():
    result = torch.export.export(Module(), (query, cache, start_pos))
print(result)

class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: "f16[1, 1, 16, 128]", arg1_1: "f16[1, 128, 16, 128]", arg2_1: "i64[1]"):
            # File: /data/users/ezyang/a/pytorch/nz.py:10 in forward, code: sp = start_pos.item()
            _local_scalar_dense: "Sym(u4)" = torch.ops.aten._local_scalar_dense.default(arg2_1);  arg2_1 = None

            # File: /data/users/ezyang/a/pytorch/torch/_export/pass_base.py:54 in _create_dummy_node_metadata, code: return NodeMetadata({"stack_trace": "".join(traceback.format_stack(limit=1))})
            ge: "Sym(u4 >= 0)" = _local_scalar_dense >= 0
            scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(ge);  ge = None
            _assert_async = torch.ops.aten._assert_async.msg(scalar_tensor, '_local_scalar_dense is outside of inline constraint [0, 126].');  scalar_tensor = None
            le: "Sym(u4 <= 126)" = _local_scalar_dense <= 126
            scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(le);  le = None
            _assert_async_1 = torch.ops.aten._assert_async.msg(scalar_tensor_1, '_local_scalar_dense is outside of inline constraint [0, 126].');  scalar_tensor_1 = None

            # File: /data/users/ezyang/a/pytorch/torch/__init__.py:2052 in _constrain_as_size, code: torch.sym_constrain_range_for_size(symbol, min=min, max=max)
            sym_constrain_range_for_size = torch.ops.aten.sym_constrain_range_for_size.default(_local_scalar_dense, min = 0, max = 126)

            # File: /data/users/ezyang/a/pytorch/nz.py:12 in forward, code: key = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
            slice_1: "f16[1, 128, 16, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807)
            add: "Sym(u4 + 1)" = _local_scalar_dense + 1
            slice_2: "f16[1, u4 + 1, 16, 128]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, add);  slice_1 = add = None
            slice_3: "f16[1, u4 + 1, 16, 128]" = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807)
            slice_4: "f16[1, u4 + 1, 16, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None

            # File: /data/users/ezyang/a/pytorch/nz.py:13 in forward, code: value = cache[:, : sp + 1, :, :]  # 1, sp+1, 16, 128
            slice_5: "f16[1, 128, 16, 128]" = torch.ops.aten.slice.Tensor(arg1_1, 0, 0, 9223372036854775807);  arg1_1 = None
            add_1: "Sym(u4 + 1)" = _local_scalar_dense + 1;  _local_scalar_dense = None
            slice_6: "f16[1, u4 + 1, 16, 128]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, add_1);  slice_5 = add_1 = None
            slice_7: "f16[1, u4 + 1, 16, 128]" = torch.ops.aten.slice.Tensor(slice_6, 2, 0, 9223372036854775807)
            slice_8: "f16[1, u4 + 1, 16, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 9223372036854775807);  slice_7 = None

            # File: /data/users/ezyang/a/pytorch/nz.py:14 in forward, code: query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            transpose: "f16[1, 16, 1, 128]" = torch.ops.aten.transpose.int(arg0_1, 1, 2);  arg0_1 = None

            # File: /data/users/ezyang/a/pytorch/nz.py:15 in forward, code: key = key.transpose(1, 2)
            transpose_1: "f16[1, 16, u4 + 1, 128]" = torch.ops.aten.transpose.int(slice_4, 1, 2);  slice_4 = None

            # File: /data/users/ezyang/a/pytorch/nz.py:16 in forward, code: value = value.transpose(1, 2)
            transpose_2: "f16[1, 16, u4 + 1, 128]" = torch.ops.aten.transpose.int(slice_8, 1, 2);  slice_8 = None

            # File: /data/users/ezyang/a/pytorch/nz.py:18 in forward, code: return scaled_dot_product_attention(query, key, value)
            mul: "f16[1, 16, 1, 128]" = torch.ops.aten.mul.Scalar(transpose, 0.29730177875068026);  transpose = None
            transpose_3: "f16[1, 16, 128, u4 + 1]" = torch.ops.aten.transpose.int(transpose_1, -2, -1);  transpose_1 = None
            mul_1: "f16[1, 16, 128, u4 + 1]" = torch.ops.aten.mul.Scalar(transpose_3, 0.29730177875068026);  transpose_3 = None
            expand: "f16[1, 16, 1, 128]" = torch.ops.aten.expand.default(mul, [1, 16, 1, 128]);  mul = None
            view: "f16[16, 1, 128]" = torch.ops.aten.view.default(expand, [16, 1, 128]);  expand = None
            sym_size: "Sym(u4 + 1)" = torch.ops.aten.sym_size.int(slice_2, 1);  slice_2 = None
            expand_1: "f16[1, 16, 128, u4 + 1]" = torch.ops.aten.expand.default(mul_1, [1, 16, 128, sym_size]);  mul_1 = None
            view_1: "f16[16, 128, u4 + 1]" = torch.ops.aten.view.default(expand_1, [16, 128, sym_size]);  expand_1 = None
            bmm: "f16[16, 1, u4 + 1]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
            view_2: "f16[1, 16, 1, u4 + 1]" = torch.ops.aten.view.default(bmm, [1, 16, 1, sym_size]);  bmm = None
            _softmax: "f16[1, 16, 1, u4 + 1]" = torch.ops.aten._softmax.default(view_2, -1, False);  view_2 = None
            expand_2: "f16[1, 16, 1, u4 + 1]" = torch.ops.aten.expand.default(_softmax, [1, 16, 1, sym_size]);  _softmax = None
            view_3: "f16[16, 1, u4 + 1]" = torch.ops.aten.view.default(expand_2, [16, 1, sym_size]);  expand_2 = sym_size = None
            sym_size_1: "Sym(u4 + 1)" = torch.ops.aten.sym_size.int(slice_6, 1);  slice_6 = None
            expand_3: "f16[1, 16, u4 + 1, 128]" = torch.ops.aten.expand.default(transpose_2, [1, 16, sym_size_1, 128]);  transpose_2 = None
            view_4: "f16[16, u4 + 1, 128]" = torch.ops.aten.view.default(expand_3, [16, sym_size_1, 128]);  expand_3 = sym_size_1 = None
            bmm_1: "f16[16, 1, 128]" = torch.ops.aten.bmm.default(view_3, view_4);  view_3 = view_4 = None
            view_5: "f16[1, 16, 1, 128]" = torch.ops.aten.view.default(bmm_1, [1, 16, 1, 128]);  bmm_1 = None
            return (view_5,)