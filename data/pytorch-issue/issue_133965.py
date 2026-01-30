import torch.nn as nn

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch._functorch.config.unlift_effect_tokens = True



isolate_fails_code_str = None



# torch version: 2.4.0+cu121
# torch cuda version: 12.1
# torch git version: e4ee3be4063b7c430974252fdf7db42273388d86


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Tue_Aug_15_22:02:13_PDT_2023 
# Cuda compilation tools, release 12.2, V12.2.140 
# Build cuda_12.2.r12.2/compiler.33191640_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_26, sym_size_int_1, primals_5, primals_6, primals_11, primals_12, primals_17, primals_18, primals_23, primals_27, addmm, addmm_1, select_2, select_3, full, exp, index_4, div, full_2, scatter_add_1, mul_5, index_7, add_6, exp_1, index_10, div_1, scatter_add_3, mul_11, index_13, add_11, exp_2, index_16, div_2, scatter_add_5, mul_17, index_19, add_16, exp_3, index_22, div_3, full_11, permute_8, permute_12, permute_16, permute_20, permute_24, permute_28, tangents_1):
        sum_5 = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True)
        view_32 = torch.ops.aten.view.default(sum_5, [2]);  sum_5 = None
        view_33 = torch.ops.aten.view.default(tangents_1, [primals_26, 1, 2]);  tangents_1 = None
        view_5 = torch.ops.aten.view.default(select_2, [-1, 1, 1])
        expand_11 = torch.ops.aten.expand.default(view_5, [sym_size_int_1, 1, 2])
        gather = torch.ops.aten.gather.default(view_33, 0, expand_11);  view_33 = expand_11 = None
        mul_21 = torch.ops.aten.mul.Tensor(gather, index_19);  index_19 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(div_3, -1)
        mul_22 = torch.ops.aten.mul.Tensor(gather, unsqueeze_3);  gather = unsqueeze_3 = None
        sum_6 = torch.ops.aten.sum.dim_IntList(mul_21, [2], True);  mul_21 = None
        squeeze = torch.ops.aten.squeeze.dim(sum_6, -1);  sum_6 = None
        index_put = torch.ops.aten.index_put.default(full_11, [select_3], mul_22, True);  mul_22 = None
        div_5 = torch.ops.aten.div.Tensor(div_3, index_22);  div_3 = None
        neg = torch.ops.aten.neg.default(squeeze)
        mul_23 = torch.ops.aten.mul.Tensor(neg, div_5);  neg = div_5 = None
        div_6 = torch.ops.aten.div.Tensor(squeeze, index_22);  squeeze = index_22 = None
        index_put_1 = torch.ops.aten.index_put.default(full, [select_2], mul_23, True);  mul_23 = None
        view_3 = torch.ops.aten.view.default(select_2, [-1, 1])
        expand = torch.ops.aten.expand.default(view_3, [sym_size_int_1, 1]);  view_3 = None
        gather_1 = torch.ops.aten.gather.default(index_put_1, 0, expand);  index_put_1 = None
        add_19 = torch.ops.aten.add.Tensor(div_6, gather_1);  div_6 = gather_1 = None
        mul_24 = torch.ops.aten.mul.Tensor(add_19, exp_3);  add_19 = exp_3 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        expand_12 = torch.ops.aten.expand.default(unsqueeze_4, [sym_size_int_1, 1, 2]);  unsqueeze_4 = None
        gt_3 = torch.ops.aten.gt.Scalar(add_16, 0)
        mul_18 = torch.ops.aten.mul.Tensor(add_16, 0.2)
        where_3 = torch.ops.aten.where.self(gt_3, add_16, mul_18);  add_16 = mul_18 = None
        mul_25 = torch.ops.aten.mul.Tensor(expand_12, where_3);  where_3 = None
        mul_26 = torch.ops.aten.mul.Tensor(expand_12, primals_23);  expand_12 = primals_23 = None
        sum_7 = torch.ops.aten.sum.dim_IntList(mul_25, [0], True);  mul_25 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_26, 0.2)
        where_4 = torch.ops.aten.where.self(gt_3, mul_26, mul_27);  gt_3 = mul_26 = mul_27 = None
        index_put_2 = torch.ops.aten.index_put.default(full_11, [select_2], where_4, True)
        index_put_3 = torch.ops.aten.index_put.default(full_11, [select_3], where_4, True);  full_11 = where_4 = None
        add_20 = torch.ops.aten.add.Tensor(index_put, index_put_3);  index_put = index_put_3 = None
        view_34 = torch.ops.aten.view.default(index_put_2, [primals_26, 2]);  index_put_2 = None
        mm = torch.ops.aten.mm.default(view_34, permute_8);  permute_8 = None
        permute_9 = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_9, mul_17);  permute_9 = None
        permute_10 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_8 = torch.ops.aten.sum.dim_IntList(view_34, [0], True);  view_34 = None
        view_35 = torch.ops.aten.view.default(sum_8, [2]);  sum_8 = None
        permute_11 = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        view_36 = torch.ops.aten.view.default(add_20, [primals_26, 2]);  add_20 = None
        mm_2 = torch.ops.aten.mm.default(view_36, permute_12);  permute_12 = None
        permute_13 = torch.ops.aten.permute.default(view_36, [1, 0])
        mm_3 = torch.ops.aten.mm.default(permute_13, mul_17);  permute_13 = mul_17 = None
        permute_14 = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
        sum_9 = torch.ops.aten.sum.dim_IntList(view_36, [0], True);  view_36 = None
        view_37 = torch.ops.aten.view.default(sum_9, [2]);  sum_9 = None
        add_21 = torch.ops.aten.add.Tensor(mm, mm_2);  mm = mm_2 = None
        permute_15 = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        view_23 = torch.ops.aten.view.default(scatter_add_5, [-1, 4]);  scatter_add_5 = None
        add_13 = torch.ops.aten.add.Tensor(view_23, primals_18);  view_23 = primals_18 = None
        mul_16 = torch.ops.aten.mul.Tensor(add_13, 0.7071067811865476)
        erf_2 = torch.ops.aten.erf.default(mul_16);  mul_16 = None
        add_14 = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_29 = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
        mul_30 = torch.ops.aten.mul.Tensor(add_13, add_13)
        mul_31 = torch.ops.aten.mul.Tensor(mul_30, -0.5);  mul_30 = None
        exp_4 = torch.ops.aten.exp.default(mul_31);  mul_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_33 = torch.ops.aten.mul.Tensor(add_13, mul_32);  add_13 = mul_32 = None
        add_23 = torch.ops.aten.add.Tensor(mul_29, mul_33);  mul_29 = mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(add_21, add_23);  add_21 = add_23 = None
        sum_10 = torch.ops.aten.sum.dim_IntList(mul_34, [0], True)
        view_38 = torch.ops.aten.view.default(sum_10, [4]);  sum_10 = None
        view_39 = torch.ops.aten.view.default(mul_34, [primals_26, 1, 4]);  mul_34 = None
        expand_2 = torch.ops.aten.expand.default(view_5, [sym_size_int_1, 1, 4]);  view_5 = None
        gather_2 = torch.ops.aten.gather.default(view_39, 0, expand_2);  view_39 = None
        mul_35 = torch.ops.aten.mul.Tensor(gather_2, index_13);  index_13 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(div_2, -1)
        mul_36 = torch.ops.aten.mul.Tensor(gather_2, unsqueeze_2);  gather_2 = unsqueeze_2 = None
        sum_11 = torch.ops.aten.sum.dim_IntList(mul_35, [2], True);  mul_35 = None
        squeeze_1 = torch.ops.aten.squeeze.dim(sum_11, -1);  sum_11 = None
        index_put_4 = torch.ops.aten.index_put.default(full_2, [select_3], mul_36, True);  mul_36 = None
        div_8 = torch.ops.aten.div.Tensor(div_2, index_16);  div_2 = None
        neg_1 = torch.ops.aten.neg.default(squeeze_1)
        mul_37 = torch.ops.aten.mul.Tensor(neg_1, div_8);  neg_1 = div_8 = None
        div_9 = torch.ops.aten.div.Tensor(squeeze_1, index_16);  squeeze_1 = index_16 = None
        index_put_5 = torch.ops.aten.index_put.default(full, [select_2], mul_37, True);  mul_37 = None
        gather_3 = torch.ops.aten.gather.default(index_put_5, 0, expand);  index_put_5 = None
        add_24 = torch.ops.aten.add.Tensor(div_9, gather_3);  div_9 = gather_3 = None
        mul_38 = torch.ops.aten.mul.Tensor(add_24, exp_2);  add_24 = exp_2 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(mul_38, -1);  mul_38 = None
        expand_13 = torch.ops.aten.expand.default(unsqueeze_5, [sym_size_int_1, 1, 4]);  unsqueeze_5 = None
        gt_2 = torch.ops.aten.gt.Scalar(add_11, 0)
        mul_12 = torch.ops.aten.mul.Tensor(add_11, 0.2)
        where_2 = torch.ops.aten.where.self(gt_2, add_11, mul_12);  add_11 = mul_12 = None
        mul_39 = torch.ops.aten.mul.Tensor(expand_13, where_2);  where_2 = None
        mul_40 = torch.ops.aten.mul.Tensor(expand_13, primals_17);  expand_13 = primals_17 = None
        sum_12 = torch.ops.aten.sum.dim_IntList(mul_39, [0], True);  mul_39 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_40, 0.2)
        where_5 = torch.ops.aten.where.self(gt_2, mul_40, mul_41);  gt_2 = mul_40 = mul_41 = None
        index_put_6 = torch.ops.aten.index_put.default(full_2, [select_2], where_5, True)
        index_put_7 = torch.ops.aten.index_put.default(full_2, [select_3], where_5, True);  where_5 = None
        add_25 = torch.ops.aten.add.Tensor(index_put_4, index_put_7);  index_put_4 = index_put_7 = None
        view_40 = torch.ops.aten.view.default(index_put_6, [primals_26, 4]);  index_put_6 = None
        mm_4 = torch.ops.aten.mm.default(view_40, permute_16);  permute_16 = None
        permute_17 = torch.ops.aten.permute.default(view_40, [1, 0])
        mm_5 = torch.ops.aten.mm.default(permute_17, mul_11);  permute_17 = None
        permute_18 = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
        sum_13 = torch.ops.aten.sum.dim_IntList(view_40, [0], True);  view_40 = None
        view_41 = torch.ops.aten.view.default(sum_13, [4]);  sum_13 = None
        permute_19 = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        view_42 = torch.ops.aten.view.default(add_25, [primals_26, 4]);  add_25 = None
        mm_6 = torch.ops.aten.mm.default(view_42, permute_20);  permute_20 = None
        permute_21 = torch.ops.aten.permute.default(view_42, [1, 0])
        mm_7 = torch.ops.aten.mm.default(permute_21, mul_11);  permute_21 = mul_11 = None
        permute_22 = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
        sum_14 = torch.ops.aten.sum.dim_IntList(view_42, [0], True);  view_42 = None
        view_43 = torch.ops.aten.view.default(sum_14, [4]);  sum_14 = None
        add_26 = torch.ops.aten.add.Tensor(mm_4, mm_6);  mm_4 = mm_6 = None
        permute_23 = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        view_15 = torch.ops.aten.view.default(scatter_add_3, [-1, 4]);  scatter_add_3 = None
        add_8 = torch.ops.aten.add.Tensor(view_15, primals_12);  view_15 = primals_12 = None
        mul_10 = torch.ops.aten.mul.Tensor(add_8, 0.7071067811865476)
        erf_1 = torch.ops.aten.erf.default(mul_10);  mul_10 = None
        add_9 = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_43 = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_44 = torch.ops.aten.mul.Tensor(add_8, add_8)
        mul_45 = torch.ops.aten.mul.Tensor(mul_44, -0.5);  mul_44 = None
        exp_5 = torch.ops.aten.exp.default(mul_45);  mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_47 = torch.ops.aten.mul.Tensor(add_8, mul_46);  add_8 = mul_46 = None
        add_28 = torch.ops.aten.add.Tensor(mul_43, mul_47);  mul_43 = mul_47 = None
        mul_48 = torch.ops.aten.mul.Tensor(add_26, add_28);  add_26 = add_28 = None
        sum_15 = torch.ops.aten.sum.dim_IntList(mul_48, [0], True)
        view_44 = torch.ops.aten.view.default(sum_15, [4]);  sum_15 = None
        view_45 = torch.ops.aten.view.default(mul_48, [primals_26, 1, 4]);  mul_48 = None
        gather_4 = torch.ops.aten.gather.default(view_45, 0, expand_2);  view_45 = None
        mul_49 = torch.ops.aten.mul.Tensor(gather_4, index_7);  index_7 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(div_1, -1)
        mul_50 = torch.ops.aten.mul.Tensor(gather_4, unsqueeze_1);  gather_4 = unsqueeze_1 = None
        sum_16 = torch.ops.aten.sum.dim_IntList(mul_49, [2], True);  mul_49 = None
        squeeze_2 = torch.ops.aten.squeeze.dim(sum_16, -1);  sum_16 = None
        index_put_8 = torch.ops.aten.index_put.default(full_2, [select_3], mul_50, True);  mul_50 = None
        div_11 = torch.ops.aten.div.Tensor(div_1, index_10);  div_1 = None
        neg_2 = torch.ops.aten.neg.default(squeeze_2)
        mul_51 = torch.ops.aten.mul.Tensor(neg_2, div_11);  neg_2 = div_11 = None
        div_12 = torch.ops.aten.div.Tensor(squeeze_2, index_10);  squeeze_2 = index_10 = None
        index_put_9 = torch.ops.aten.index_put.default(full, [select_2], mul_51, True);  mul_51 = None
        gather_5 = torch.ops.aten.gather.default(index_put_9, 0, expand);  index_put_9 = None
        add_29 = torch.ops.aten.add.Tensor(div_12, gather_5);  div_12 = gather_5 = None
        mul_52 = torch.ops.aten.mul.Tensor(add_29, exp_1);  add_29 = exp_1 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(mul_52, -1);  mul_52 = None
        expand_14 = torch.ops.aten.expand.default(unsqueeze_6, [sym_size_int_1, 1, 4]);  unsqueeze_6 = None
        gt_1 = torch.ops.aten.gt.Scalar(add_6, 0)
        mul_6 = torch.ops.aten.mul.Tensor(add_6, 0.2)
        where_1 = torch.ops.aten.where.self(gt_1, add_6, mul_6);  add_6 = mul_6 = None
        mul_53 = torch.ops.aten.mul.Tensor(expand_14, where_1);  where_1 = None
        mul_54 = torch.ops.aten.mul.Tensor(expand_14, primals_11);  expand_14 = primals_11 = None
        sum_17 = torch.ops.aten.sum.dim_IntList(mul_53, [0], True);  mul_53 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_54, 0.2)
        where_6 = torch.ops.aten.where.self(gt_1, mul_54, mul_55);  gt_1 = mul_54 = mul_55 = None
        index_put_10 = torch.ops.aten.index_put.default(full_2, [select_2], where_6, True)
        index_put_11 = torch.ops.aten.index_put.default(full_2, [select_3], where_6, True);  where_6 = None
        add_30 = torch.ops.aten.add.Tensor(index_put_8, index_put_11);  index_put_8 = index_put_11 = None
        view_46 = torch.ops.aten.view.default(index_put_10, [primals_26, 4]);  index_put_10 = None
        mm_8 = torch.ops.aten.mm.default(view_46, permute_24);  permute_24 = None
        permute_25 = torch.ops.aten.permute.default(view_46, [1, 0])
        mm_9 = torch.ops.aten.mm.default(permute_25, mul_5);  permute_25 = None
        permute_26 = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
        sum_18 = torch.ops.aten.sum.dim_IntList(view_46, [0], True);  view_46 = None
        view_47 = torch.ops.aten.view.default(sum_18, [4]);  sum_18 = None
        permute_27 = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        view_48 = torch.ops.aten.view.default(add_30, [primals_26, 4]);  add_30 = None
        mm_10 = torch.ops.aten.mm.default(view_48, permute_28);  permute_28 = None
        permute_29 = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_11 = torch.ops.aten.mm.default(permute_29, mul_5);  permute_29 = mul_5 = None
        permute_30 = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
        sum_19 = torch.ops.aten.sum.dim_IntList(view_48, [0], True);  view_48 = None
        view_49 = torch.ops.aten.view.default(sum_19, [4]);  sum_19 = None
        add_31 = torch.ops.aten.add.Tensor(mm_8, mm_10);  mm_8 = mm_10 = None
        permute_31 = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        view_7 = torch.ops.aten.view.default(scatter_add_1, [-1, 4]);  scatter_add_1 = None
        add_3 = torch.ops.aten.add.Tensor(view_7, primals_6);  view_7 = primals_6 = None
        mul_4 = torch.ops.aten.mul.Tensor(add_3, 0.7071067811865476)
        erf = torch.ops.aten.erf.default(mul_4);  mul_4 = None
        add_4 = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_57 = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_58 = torch.ops.aten.mul.Tensor(add_3, add_3)
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, -0.5);  mul_58 = None
        exp_6 = torch.ops.aten.exp.default(mul_59);  mul_59 = None
        mul_60 = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_61 = torch.ops.aten.mul.Tensor(add_3, mul_60);  add_3 = mul_60 = None
        add_33 = torch.ops.aten.add.Tensor(mul_57, mul_61);  mul_57 = mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_31, add_33);  add_31 = add_33 = None
        sum_20 = torch.ops.aten.sum.dim_IntList(mul_62, [0], True)
        view_50 = torch.ops.aten.view.default(sum_20, [4]);  sum_20 = None
        view_51 = torch.ops.aten.view.default(mul_62, [primals_26, 1, 4]);  mul_62 = None
        gather_6 = torch.ops.aten.gather.default(view_51, 0, expand_2);  view_51 = expand_2 = None
        view = torch.ops.aten.view.default(addmm, [-1, 1, 4]);  addmm = None
        index_1 = torch.ops.aten.index.Tensor(view, [select_3]);  view = None
        mul_63 = torch.ops.aten.mul.Tensor(gather_6, index_1)
        unsqueeze = torch.ops.aten.unsqueeze.default(div, -1)
        mul_64 = torch.ops.aten.mul.Tensor(gather_6, unsqueeze);  gather_6 = unsqueeze = None
        sum_21 = torch.ops.aten.sum.dim_IntList(mul_63, [2], True);  mul_63 = None
        squeeze_3 = torch.ops.aten.squeeze.dim(sum_21, -1);  sum_21 = None
        index_put_12 = torch.ops.aten.index_put.default(full_2, [select_3], mul_64, True);  mul_64 = None
        div_14 = torch.ops.aten.div.Tensor(div, index_4);  div = None
        neg_3 = torch.ops.aten.neg.default(squeeze_3)
        mul_65 = torch.ops.aten.mul.Tensor(neg_3, div_14);  neg_3 = div_14 = None
        div_15 = torch.ops.aten.div.Tensor(squeeze_3, index_4);  squeeze_3 = index_4 = None
        index_put_13 = torch.ops.aten.index_put.default(full, [select_2], mul_65, True);  full = mul_65 = None
        gather_7 = torch.ops.aten.gather.default(index_put_13, 0, expand);  index_put_13 = expand = None
        add_34 = torch.ops.aten.add.Tensor(div_15, gather_7);  div_15 = gather_7 = None
        mul_66 = torch.ops.aten.mul.Tensor(add_34, exp);  add_34 = exp = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
        expand_15 = torch.ops.aten.expand.default(unsqueeze_7, [sym_size_int_1, 1, 4]);  unsqueeze_7 = sym_size_int_1 = None
        view_1 = torch.ops.aten.view.default(addmm_1, [-1, 1, 4]);  addmm_1 = None
        index_2 = torch.ops.aten.index.Tensor(view_1, [select_2]);  view_1 = None
        add_1 = torch.ops.aten.add.Tensor(index_2, index_1);  index_2 = index_1 = None
        gt = torch.ops.aten.gt.Scalar(add_1, 0)
        mul = torch.ops.aten.mul.Tensor(add_1, 0.2)
        where = torch.ops.aten.where.self(gt, add_1, mul);  add_1 = mul = None
        mul_67 = torch.ops.aten.mul.Tensor(expand_15, where);  where = None
        mul_68 = torch.ops.aten.mul.Tensor(expand_15, primals_5);  expand_15 = primals_5 = None
        sum_22 = torch.ops.aten.sum.dim_IntList(mul_67, [0], True);  mul_67 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_68, 0.2)
        where_7 = torch.ops.aten.where.self(gt, mul_68, mul_69);  gt = mul_68 = mul_69 = None
        index_put_14 = torch.ops.aten.index_put.default(full_2, [select_2], where_7, True);  select_2 = None
        index_put_15 = torch.ops.aten.index_put.default(full_2, [select_3], where_7, True);  full_2 = select_3 = where_7 = None
        add_35 = torch.ops.aten.add.Tensor(index_put_12, index_put_15);  index_put_12 = index_put_15 = None
        view_52 = torch.ops.aten.view.default(index_put_14, [primals_26, 4]);  index_put_14 = None
        permute_32 = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_12 = torch.ops.aten.mm.default(permute_32, primals_27);  permute_32 = None
        permute_33 = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
        sum_23 = torch.ops.aten.sum.dim_IntList(view_52, [0], True);  view_52 = None
        view_53 = torch.ops.aten.view.default(sum_23, [4]);  sum_23 = None
        permute_34 = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        view_54 = torch.ops.aten.view.default(add_35, [primals_26, 4]);  add_35 = primals_26 = None
        permute_35 = torch.ops.aten.permute.default(view_54, [1, 0])
        mm_13 = torch.ops.aten.mm.default(permute_35, primals_27);  permute_35 = primals_27 = None
        permute_36 = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
        sum_24 = torch.ops.aten.sum.dim_IntList(view_54, [0], True);  view_54 = None
        view_55 = torch.ops.aten.view.default(sum_24, [4]);  sum_24 = None
        permute_37 = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        return [permute_37, view_55, permute_34, view_53, sum_22, view_50, permute_31, view_49, permute_27, view_47, sum_17, view_44, permute_23, view_43, permute_19, view_41, sum_12, view_38, permute_15, view_37, permute_11, view_35, sum_7, view_32, None, None, None, None, None, None, None, None]
        
def load_args(reader):
    reader.symint(900)  # primals_26
    reader.symint(None)  # sym_size_int_1
    buf0 = reader.storage(None, 16)
    reader.tensor(buf0, (1, 1, 4), is_leaf=True)  # primals_5
    buf1 = reader.storage(None, 16)
    reader.tensor(buf1, (4,), is_leaf=True)  # primals_6
    buf2 = reader.storage(None, 16)
    reader.tensor(buf2, (1, 1, 4), is_leaf=True)  # primals_11
    buf3 = reader.storage(None, 16)
    reader.tensor(buf3, (4,), is_leaf=True)  # primals_12
    buf4 = reader.storage(None, 16)
    reader.tensor(buf4, (1, 1, 4), is_leaf=True)  # primals_17
    buf5 = reader.storage(None, 16)
    reader.tensor(buf5, (4,), is_leaf=True)  # primals_18
    buf6 = reader.storage(None, 8)
    reader.tensor(buf6, (1, 1, 2), is_leaf=True)  # primals_23
    buf7 = reader.storage(None, 8*s1)
    reader.tensor(buf7, (s1, 2), is_leaf=True)  # primals_27
    buf8 = reader.storage(None, 16*s1)
    reader.tensor(buf8, (s1, 4), is_leaf=True)  # addmm
    buf9 = reader.storage(None, 16*s1)
    reader.tensor(buf9, (s1, 4), is_leaf=True)  # addmm_1
    buf10 = reader.storage(None, 16*s1 + 16*u0, dtype_hint=torch.int64)
    reader.tensor(buf10, (s1 + u0,), dtype=torch.int64, storage_offset=Max(1, s1 + u0), is_leaf=True)  # select_2
    reader.tensor(buf10, (s1 + u0,), dtype=torch.int64, is_leaf=True)  # select_3
    buf11 = reader.storage(None, 4*s1)
    reader.tensor(buf11, (s1, 1), is_leaf=True)  # full
    buf12 = reader.storage(None, 4*s1 + 4*u0)
    reader.tensor(buf12, (s1 + u0, 1), is_leaf=True)  # exp
    buf13 = reader.storage(None, 4*s1 + 4*u0)
    reader.tensor(buf13, (s1 + u0, 1), is_leaf=True)  # index_4
    buf14 = reader.storage(None, 4*s1 + 4*u0)
    reader.tensor(buf14, (s1 + u0, 1), is_leaf=True)  # div
    buf15 = reader.storage(None, 16*s1)
    reader.tensor(buf15, (s1, 1, 4), is_leaf=True)  # full_2
    buf16 = reader.storage(None, 16*s1)
    reader.tensor(buf16, (s1, 1, 4), is_leaf=True)  # scatter_add_1
    buf17 = reader.storage(None, 16*s1)
    reader.tensor(buf17, (s1, 4), is_leaf=True)  # mul_5
    buf18 = reader.storage(None, 16*s1 + 16*u1)
    reader.tensor(buf18, (s1 + u1, 1, 4), is_leaf=True)  # index_7
    buf19 = reader.storage(None, 16*s1 + 16*u1)
    reader.tensor(buf19, (s1 + u1, 1, 4), is_leaf=True)  # add_6
    buf20 = reader.storage(None, 4*s1 + 4*u1)
    reader.tensor(buf20, (s1 + u1, 1), is_leaf=True)  # exp_1
    buf21 = reader.storage(None, 4*s1 + 4*u1)
    reader.tensor(buf21, (s1 + u1, 1), is_leaf=True)  # index_10
    buf22 = reader.storage(None, 4*s1 + 4*u1)
    reader.tensor(buf22, (s1 + u1, 1), is_leaf=True)  # div_1
    buf23 = reader.storage(None, 16*s1)
    reader.tensor(buf23, (s1, 1, 4), is_leaf=True)  # scatter_add_3
    buf24 = reader.storage(None, 16*s1)
    reader.tensor(buf24, (s1, 4), is_leaf=True)  # mul_11
    buf25 = reader.storage(None, 16*s1 + 16*u2)
    reader.tensor(buf25, (s1 + u2, 1, 4), is_leaf=True)  # index_13
    buf26 = reader.storage(None, 16*s1 + 16*u2)
    reader.tensor(buf26, (s1 + u2, 1, 4), is_leaf=True)  # add_11
    buf27 = reader.storage(None, 4*s1 + 4*u2)
    reader.tensor(buf27, (s1 + u2, 1), is_leaf=True)  # exp_2
    buf28 = reader.storage(None, 4*s1 + 4*u2)
    reader.tensor(buf28, (s1 + u2, 1), is_leaf=True)  # index_16
    buf29 = reader.storage(None, 4*s1 + 4*u2)
    reader.tensor(buf29, (s1 + u2, 1), is_leaf=True)  # div_2
    buf30 = reader.storage(None, 16*s1)
    reader.tensor(buf30, (s1, 1, 4), is_leaf=True)  # scatter_add_5
    buf31 = reader.storage(None, 16*s1)
    reader.tensor(buf31, (s1, 4), is_leaf=True)  # mul_17
    buf32 = reader.storage(None, 8*s1 + 8*u3)
    reader.tensor(buf32, (s1 + u3, 1, 2), is_leaf=True)  # index_19
    buf33 = reader.storage(None, 8*s1 + 8*u3)
    reader.tensor(buf33, (s1 + u3, 1, 2), is_leaf=True)  # add_16
    buf34 = reader.storage(None, 4*s1 + 4*u3)
    reader.tensor(buf34, (s1 + u3, 1), is_leaf=True)  # exp_3
    buf35 = reader.storage(None, 4*s1 + 4*u3)
    reader.tensor(buf35, (s1 + u3, 1), is_leaf=True)  # index_22
    buf36 = reader.storage(None, 4*s1 + 4*u3)
    reader.tensor(buf36, (s1 + u3, 1), is_leaf=True)  # div_3
    buf37 = reader.storage(None, 8*s1)
    reader.tensor(buf37, (s1, 1, 2), is_leaf=True)  # full_11
    buf38 = reader.storage(None, 32)
    reader.tensor(buf38, (2, 4), is_leaf=True)  # permute_8
    buf39 = reader.storage(None, 32)
    reader.tensor(buf39, (2, 4), is_leaf=True)  # permute_12
    buf40 = reader.storage(None, 64)
    reader.tensor(buf40, (4, 4), is_leaf=True)  # permute_16
    buf41 = reader.storage(None, 64)
    reader.tensor(buf41, (4, 4), is_leaf=True)  # permute_20
    buf42 = reader.storage(None, 64)
    reader.tensor(buf42, (4, 4), is_leaf=True)  # permute_24
    buf43 = reader.storage(None, 64)
    reader.tensor(buf43, (4, 4), is_leaf=True)  # permute_28
    buf44 = reader.storage(None, 8*s1)
    reader.tensor(buf44, (s1, 2), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='minify', save_dir='/home/usevitch/code/python/neurosub/torch_compile_debug/run_2024_08_19_22_23_52_780264-pid_1226344/minifier/checkpoints', tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir='/home/usevitch/code/python/neurosub/torch_compile_debug/run_2024_08_19_22_23_52_780264-pid_1226344/minifier/checkpoints', tracing_mode='symbolic', check_str=None)
        # mod(*args)