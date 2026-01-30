import torch

def forward(self, arg0_1: f32[768], arg1_1: f32[768], arg2_1: f32[768], arg3_1: f32[768], arg4_1: f32[768], arg5_1: f32[768], arg6_1: f32[8, 576, 768], arg7_1: f32[8, 1, 768], arg8_1: f32[8, 768], arg9_1: f32[8, 577, 1], arg10_1: f32[8, 577, 1], arg11_1: f32[768, 768], arg12_1: f32[128, 1, 577], arg13_1: f32[768, 768]):
        # No stacktrace found for following nodes
        view: f32[8, 1, 768] = torch.ops.aten.view.default(arg8_1, [8, 1, 768]);  arg8_1 = None
        mul: f32[8, 1, 768] = torch.ops.aten.mul.Tensor(arg0_1, view);  arg0_1 = view = None
        add: f32[8, 1, 768] = torch.ops.aten.add.Tensor(arg7_1, mul);  arg7_1 = mul = None
        cat: f32[8, 577, 768] = torch.ops.aten.cat.default([add, arg6_1], 1);  arg6_1 = None
        sub: f32[8, 577, 768] = torch.ops.aten.sub.Tensor(cat, arg9_1);  cat = arg9_1 = None
        mul_1: f32[8, 577, 768] = torch.ops.aten.mul.Tensor(sub, arg10_1);  sub = arg10_1 = None
        mul_2: f32[8, 577, 768] = torch.ops.aten.mul.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
        add_1: f32[8, 577, 768] = torch.ops.aten.add.Tensor(mul_2, arg3_1);  mul_2 = arg3_1 = None
        view_1: f32[4616, 768] = torch.ops.aten.view.default(add_1, [4616, 768]);  add_1 = None
        addmm: f32[4616, 768] = torch.ops.aten.addmm.default(arg4_1, view_1, arg11_1);  arg4_1 = view_1 = arg11_1 = None
        view_2: f32[8, 577, 768] = torch.ops.aten.view.default(addmm, [8, 577, 768]);  addmm = None
        view_3: f32[8, 577, 16, 48] = torch.ops.aten.view.default(view_2, [8, 577, 16, 48]);  view_2 = None
        permute: f32[8, 16, 577, 48] = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
        expand: f32[8, 16, 577, 48] = torch.ops.aten.expand.default(permute, [8, 16, 577, 48]);  permute = None
        clone: f32[8, 16, 577, 48] = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_4: f32[128, 577, 48] = torch.ops.aten.view.default(clone, [128, 577, 48]);  clone = None
        bmm: f32[128, 1, 48] = torch.ops.aten.bmm.default(arg12_1, view_4);  arg12_1 = view_4 = None
        view_5: f32[8, 16, 1, 48] = torch.ops.aten.view.default(bmm, [8, 16, 1, 48]);  bmm = None
        permute_1: f32[8, 1, 16, 48] = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        view_6: f32[8, 1, 768] = torch.ops.aten.view.default(permute_1, [8, 1, 768]);  permute_1 = None
        view_7: f32[8, 768] = torch.ops.aten.view.default(view_6, [8, 768]);  view_6 = None
        addmm_1: f32[8, 768] = torch.ops.aten.addmm.default(arg5_1, view_7, arg13_1);  arg5_1 = view_7 = arg13_1 = None
        view_8: f32[8, 1, 768] = torch.ops.aten.view.default(addmm_1, [8, 1, 768]);  addmm_1 = None
        mul_3: f32[8, 1, 768] = torch.ops.aten.mul.Tensor(arg1_1, view_8);  arg1_1 = view_8 = None
        add_2: f32[8, 1, 768] = torch.ops.aten.add.Tensor(add, mul_3);  add = mul_3 = None
        return (add_2,)