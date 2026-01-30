import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
                clone_50,
                gt_scalar,
                div_tensor,
                convert_element_type_default_7,
                convert_element_type_default_13,
                convert_element_type_default_14
    ):
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(clone_50, torch.float32);  clone_50 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_4, [336, 512, 64]);  convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_6, torch.bfloat16);  view_default_6 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        expand_default_2 = torch.ops.aten.expand.default(mul_tensor_1, [28, 12, 512, 512]);  mul_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(expand_default_2, [336, 512, 512]);  expand_default_2 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(permute_default_4, torch.bfloat16);  permute_default_4 = None
        bmm_default_2 = torch.ops.aten.bmm.default(convert_element_type_default_6, convert_element_type_default_5);  convert_element_type_default_6 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(bmm_default_2, torch.float32);  bmm_default_2 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_10, [28, 12, 512, 64]);  convert_element_type_default_10 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(view_default_7, torch.bfloat16);  view_default_7 = None
        permute_default_9 = torch.ops.aten.permute.default(convert_element_type_default_18, [0, 2, 1, 3]);  convert_element_type_default_18 = None
        bmm_default_3 = torch.ops.aten.bmm.default(convert_element_type_default_5, convert_element_type_default_7);  convert_element_type_default_5 = convert_element_type_default_7 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(bmm_default_3, torch.float32);  bmm_default_3 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_9, [28, 12, 512, 512]);  convert_element_type_default_9 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(gt_scalar, torch.float32);  gt_scalar = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.1111111111111112);  convert_element_type_default_11 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor_2);  view_default_8 = mul_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, div_tensor);  mul_tensor_3 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [-1], True)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list_1, mul_tensor_4);  neg_default = sum_dim_int_list_1 = mul_tensor_4 = None
        view_default_9 = torch.ops.aten.view.default(fma_default, [336, 512, 512]);  fma_default = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        bmm_default_4 = torch.ops.aten.bmm.default(convert_element_type_default_13, convert_element_type_default_12);  convert_element_type_default_13 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(bmm_default_4, torch.float32);  bmm_default_4 = None
        view_default_10 = torch.ops.aten.view.default(convert_element_type_default_17, [28, 12, 64, 512]);  convert_element_type_default_17 = None
        mul_scalar_2 = torch.ops.aten.mul.Scalar(view_default_10, 0.3535533905932738);  view_default_10 = None
        permute_default_8 = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(permute_default_8, torch.bfloat16);  permute_default_8 = None
        permute_default_10 = torch.ops.aten.permute.default(convert_element_type_default_19, [0, 2, 1, 3]);  convert_element_type_default_19 = None
        bmm_default_5 = torch.ops.aten.bmm.default(convert_element_type_default_12, convert_element_type_default_14);  convert_element_type_default_12 = convert_element_type_default_14 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(bmm_default_5, torch.float32);  bmm_default_5 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_16, [28, 12, 512, 64]);  convert_element_type_default_16 = None
        mul_scalar_3 = torch.ops.aten.mul.Scalar(view_default_11, 0.3535533905932738);  view_default_11 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(mul_scalar_3, torch.bfloat16);  mul_scalar_3 = None
        permute_default_11 = torch.ops.aten.permute.default(convert_element_type_default_20, [0, 2, 1, 3]);  convert_element_type_default_20 = None
        clone_52 = torch.ops.aten.clone.default(permute_default_11, memory_format = torch.contiguous_format);  permute_default_11 = None
        view_283 = torch.ops.aten.view.default(clone_52, [28, 512, 768]);  clone_52 = None
        clone_53 = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_284 = torch.ops.aten.view.default(clone_53, [28, 512, 768]);  clone_53 = None
        view_285 = torch.ops.aten.view.default(view_284, [14336, 768]);  view_284 = None
        return view_283, view_285
 
if __name__ == "__main__":
    mod = M().to(torch.bfloat16).eval()
 
    clone_50 = torch.randn((28, 12, 512, 64), dtype=torch.bfloat16)
    gt_scalar = torch.randint(0, 2, (28, 12, 512, 512), dtype=torch.bool)
    div_tensor = torch.randn((28, 12, 512, 512), dtype=torch.float)
    convert_element_type_default_7 = torch.randn((336, 64, 512), dtype=torch.bfloat16)
    convert_element_type_default_13 = torch.randn((336, 64, 512), dtype=torch.bfloat16)
    convert_element_type_default_14 = torch.randn((336, 512, 64), dtype=torch.bfloat16)
    inputs = (
        clone_50,
        gt_scalar,
        div_tensor,
        convert_element_type_default_7,
        convert_element_type_default_13,
        convert_element_type_default_14
    )

    with torch.cpu.amp.autocast():
        compiler_mode = torch.compile(mod)
        _ = compiler_mode(*inputs)
        output = compiler_mode(*inputs)