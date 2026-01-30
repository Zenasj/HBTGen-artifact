import torch

def forward(self, primals_1: "f32[2, 4]", primals_2: "f32[2]", primals_5: "f32[s0, s1]"):
    view: "f32[s0, 4, (s1//4)]" = torch.ops.aten.reshape.default(primals_5, [-1, 4, 128]);  primals_5 = None
    permute: "f32[s0, (s1//4), 4]" = torch.ops.aten.permute.default(view, [0, 2, 1])
    clone: "f32[s0, (s1//4), 4]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view_1: "f32[s0*((s1//4)), 4]" = torch.ops.aten.reshape.default(clone, [-1, 4]);  clone = None
    permute_1: "f32[4, 2]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
    addmm: "f32[s0*((s1//4)), 2]" = torch.ops.aten.addmm.default(primals_2, view_1, permute_1);  primals_2 = None
    view_2: "f32[((s0*((s1//4)))//128), ((s0*((s1//4)))//(((s0*((s1//4)))//128))), 2]" = torch.ops.aten.reshape.default(addmm, [-1, 128, 2]);  addmm = None
    permute_2: "f32[((s0*((s1//4)))//128), 2, ((s0*((s1//4)))//(((s0*((s1//4)))//128)))]" = torch.ops.aten.permute.default(view_2, [0, 2, 1])
    clone_1: "f32[((s0*((s1//4)))//128), 2, ((s0*((s1//4)))//(((s0*((s1//4)))//128)))]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    sym_size_int: "Sym(((s0*((s1//4)))//128))" = torch.ops.aten.sym_size.int(view_2, 0)
    sym_size_int_1: "Sym(((s0*((s1//4)))//(((s0*((s1//4)))//128))))" = torch.ops.aten.sym_size.int(view_2, 1);  view_2 = None
    mul: "Sym(2*(((s0*((s1//4)))//(((s0*((s1//4)))//128)))))" = 2 * sym_size_int_1
    view_3: "f32[((s0*((s1//4)))//128), 2*(((s0*((s1//4)))//(((s0*((s1//4)))//128))))]" = torch.ops.aten.reshape.default(clone_1, [sym_size_int, mul]);  clone_1 = mul = None    
    permute_4: "f32[2, 4]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    sym_size_int_4: "Sym((s1//4))" = torch.ops.aten.sym_size.int(view, 2);  view = None
    return [view_3, view_1, permute_4, sym_size_int, sym_size_int_1, sym_size_int_4]
        
s0 = 727828 # 727828 // 2 works
s1 = 512

args = [object(), torch.rand([2, 4], device="cuda"), torch.rand([2], device="cuda"), torch.rand([s0, s1], device="cuda")]
foo_c = torch.compile(forward)
foo_c(*args)