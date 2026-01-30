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
torch._dynamo.config.suppress_errors = True




isolate_fails_code_str = None



# torch version: 2.1.0a0+git7bcf7da
# torch git version: 7bcf7da3a268b435777fe87c7794c382f444e86d




from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104):
        clone = torch.ops.aten.clone.default(primals_104);  primals_104 = None
        convolution = torch.ops.aten.convolution.default(primals_103, primals_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
        add = torch.ops.aten.add.Tensor(primals_54, 1);  primals_54 = None
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem = var_mean[0]
        getitem_1 = var_mean[1];  var_mean = None
        add_1 = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1 = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1 = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2 = torch.ops.aten.mul.Tensor(primals_52, 0.9);  primals_52 = None
        add_2 = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2 = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(squeeze_2, 1.0001594642002871);  squeeze_2 = None
        mul_4 = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(primals_53, 0.9);  primals_53 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(primals_2, -1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6 = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(primals_3, -1);  primals_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        add_5 = torch.ops.aten.add.Tensor(clone, add_4);  clone = add_4 = None
        relu = torch.ops.aten.relu.default(add_5);  add_5 = None
        convolution_1 = torch.ops.aten.convolution.default(relu, primals_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_6 = torch.ops.aten.add.Tensor(primals_57, 1);  primals_57 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2 = var_mean_1[0]
        getitem_3 = var_mean_1[1];  var_mean_1 = None
        add_7 = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1 = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_1 = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7 = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3 = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4 = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8 = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9 = torch.ops.aten.mul.Tensor(primals_55, 0.9);  primals_55 = None
        add_8 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5 = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10 = torch.ops.aten.mul.Tensor(squeeze_5, 1.0001594642002871);  squeeze_5 = None
        mul_11 = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12 = torch.ops.aten.mul.Tensor(primals_56, 0.9);  primals_56 = None
        add_9 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(primals_5, -1)
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13 = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(primals_6, -1);  primals_6 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10 = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        relu_1 = torch.ops.aten.relu.default(add_10);  add_10 = None
        convolution_2 = torch.ops.aten.convolution.default(relu_1, primals_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_11 = torch.ops.aten.add.Tensor(primals_60, 1);  primals_60 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4 = var_mean_2[0]
        getitem_5 = var_mean_2[1];  var_mean_2 = None
        add_12 = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2 = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        sub_2 = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14 = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6 = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_7 = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15 = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16 = torch.ops.aten.mul.Tensor(primals_58, 0.9);  primals_58 = None
        add_13 = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8 = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17 = torch.ops.aten.mul.Tensor(squeeze_8, 1.0001594642002871);  squeeze_8 = None
        mul_18 = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19 = torch.ops.aten.mul.Tensor(primals_59, 0.9);  primals_59 = None
        add_14 = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8 = torch.ops.aten.unsqueeze.default(primals_8, -1)
        unsqueeze_9 = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20 = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10 = torch.ops.aten.unsqueeze.default(primals_9, -1);  primals_9 = None
        unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_15 = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        relu_2 = torch.ops.aten.relu.default(add_15);  add_15 = None
        convolution_3 = torch.ops.aten.convolution.default(relu_2, primals_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_16 = torch.ops.aten.add.Tensor(primals_63, 1);  primals_63 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_3, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6 = var_mean_3[0]
        getitem_7 = var_mean_3[1];  var_mean_3 = None
        add_17 = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3 = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_3 = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21 = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9 = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10 = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22 = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23 = torch.ops.aten.mul.Tensor(primals_61, 0.9);  primals_61 = None
        add_18 = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11 = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_24 = torch.ops.aten.mul.Tensor(squeeze_11, 1.0001594642002871);  squeeze_11 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26 = torch.ops.aten.mul.Tensor(primals_62, 0.9);  primals_62 = None
        add_19 = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12 = torch.ops.aten.unsqueeze.default(primals_11, -1)
        unsqueeze_13 = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27 = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14 = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_20 = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        add_21 = torch.ops.aten.add.Tensor(add_20, relu);  add_20 = None
        relu_3 = torch.ops.aten.relu.default(add_21);  add_21 = None
        convolution_4 = torch.ops.aten.convolution.default(relu_3, primals_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_22 = torch.ops.aten.add.Tensor(primals_66, 1);  primals_66 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_4, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8 = var_mean_4[0]
        getitem_9 = var_mean_4[1];  var_mean_4 = None
        add_23 = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4 = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
        sub_4 = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28 = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12 = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13 = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29 = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30 = torch.ops.aten.mul.Tensor(primals_64, 0.9);  primals_64 = None
        add_24 = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14 = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_31 = torch.ops.aten.mul.Tensor(squeeze_14, 1.0001594642002871);  squeeze_14 = None
        mul_32 = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33 = torch.ops.aten.mul.Tensor(primals_65, 0.9);  primals_65 = None
        add_25 = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16 = torch.ops.aten.unsqueeze.default(primals_14, -1)
        unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34 = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18 = torch.ops.aten.unsqueeze.default(primals_15, -1);  primals_15 = None
        unsqueeze_19 = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_26 = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        relu_4 = torch.ops.aten.relu.default(add_26);  add_26 = None
        convolution_5 = torch.ops.aten.convolution.default(relu_4, primals_16, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_27 = torch.ops.aten.add.Tensor(primals_69, 1);  primals_69 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10 = var_mean_5[0]
        getitem_11 = var_mean_5[1];  var_mean_5 = None
        add_28 = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5 = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_5 = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35 = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15 = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16 = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36 = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37 = torch.ops.aten.mul.Tensor(primals_67, 0.9);  primals_67 = None
        add_29 = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17 = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_38 = torch.ops.aten.mul.Tensor(squeeze_17, 1.0001594642002871);  squeeze_17 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40 = torch.ops.aten.mul.Tensor(primals_68, 0.9);  primals_68 = None
        add_30 = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20 = torch.ops.aten.unsqueeze.default(primals_17, -1)
        unsqueeze_21 = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41 = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22 = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_31 = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_5 = torch.ops.aten.relu.default(add_31);  add_31 = None
        convolution_6 = torch.ops.aten.convolution.default(relu_5, primals_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_32 = torch.ops.aten.add.Tensor(primals_72, 1);  primals_72 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12 = var_mean_6[0]
        getitem_13 = var_mean_6[1];  var_mean_6 = None
        add_33 = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6 = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
        sub_6 = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42 = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18 = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_19 = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43 = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44 = torch.ops.aten.mul.Tensor(primals_70, 0.9);  primals_70 = None
        add_34 = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20 = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_45 = torch.ops.aten.mul.Tensor(squeeze_20, 1.0001594642002871);  squeeze_20 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47 = torch.ops.aten.mul.Tensor(primals_71, 0.9);  primals_71 = None
        add_35 = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24 = torch.ops.aten.unsqueeze.default(primals_20, -1)
        unsqueeze_25 = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48 = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26 = torch.ops.aten.unsqueeze.default(primals_21, -1);  primals_21 = None
        unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_36 = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        add_37 = torch.ops.aten.add.Tensor(add_36, relu_3);  add_36 = None
        relu_6 = torch.ops.aten.relu.default(add_37);  add_37 = None
        convolution_7 = torch.ops.aten.convolution.default(relu_6, primals_22, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_38 = torch.ops.aten.add.Tensor(primals_75, 1);  primals_75 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14 = var_mean_7[0]
        getitem_15 = var_mean_7[1];  var_mean_7 = None
        add_39 = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7 = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_7 = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49 = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21 = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22 = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50 = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51 = torch.ops.aten.mul.Tensor(primals_73, 0.9);  primals_73 = None
        add_40 = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23 = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_52 = torch.ops.aten.mul.Tensor(squeeze_23, 1.0001594642002871);  squeeze_23 = None
        mul_53 = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54 = torch.ops.aten.mul.Tensor(primals_74, 0.9);  primals_74 = None
        add_41 = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28 = torch.ops.aten.unsqueeze.default(primals_23, -1)
        unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55 = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30 = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
        unsqueeze_31 = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_42 = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        relu_7 = torch.ops.aten.relu.default(add_42);  add_42 = None
        convolution_8 = torch.ops.aten.convolution.default(relu_7, primals_25, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_43 = torch.ops.aten.add.Tensor(primals_78, 1);  primals_78 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16 = var_mean_8[0]
        getitem_17 = var_mean_8[1];  var_mean_8 = None
        add_44 = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8 = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        sub_8 = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56 = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24 = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25 = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57 = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58 = torch.ops.aten.mul.Tensor(primals_76, 0.9);  primals_76 = None
        add_45 = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26 = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_59 = torch.ops.aten.mul.Tensor(squeeze_26, 1.0001594642002871);  squeeze_26 = None
        mul_60 = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61 = torch.ops.aten.mul.Tensor(primals_77, 0.9);  primals_77 = None
        add_46 = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32 = torch.ops.aten.unsqueeze.default(primals_26, -1)
        unsqueeze_33 = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62 = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34 = torch.ops.aten.unsqueeze.default(primals_27, -1);  primals_27 = None
        unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_47 = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        relu_8 = torch.ops.aten.relu.default(add_47);  add_47 = None
        convolution_9 = torch.ops.aten.convolution.default(relu_8, primals_28, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_48 = torch.ops.aten.add.Tensor(primals_81, 1);  primals_81 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_9, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18 = var_mean_9[0]
        getitem_19 = var_mean_9[1];  var_mean_9 = None
        add_49 = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9 = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_9 = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_63 = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27 = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_28 = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64 = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65 = torch.ops.aten.mul.Tensor(primals_79, 0.9);  primals_79 = None
        add_50 = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29 = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_66 = torch.ops.aten.mul.Tensor(squeeze_29, 1.0001594642002871);  squeeze_29 = None
        mul_67 = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68 = torch.ops.aten.mul.Tensor(primals_80, 0.9);  primals_80 = None
        add_51 = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36 = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_37 = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69 = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38 = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
        unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_52 = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        add_53 = torch.ops.aten.add.Tensor(add_52, relu_6);  add_52 = None
        relu_9 = torch.ops.aten.relu.default(add_53);  add_53 = None
        convolution_10 = torch.ops.aten.convolution.default(relu_9, primals_31, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_54 = torch.ops.aten.add.Tensor(primals_84, 1);  primals_84 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_10, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20 = var_mean_10[0]
        getitem_21 = var_mean_10[1];  var_mean_10 = None
        add_55 = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10 = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
        sub_10 = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70 = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30 = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31 = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71 = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72 = torch.ops.aten.mul.Tensor(primals_82, 0.9);  primals_82 = None
        add_56 = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32 = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_73 = torch.ops.aten.mul.Tensor(squeeze_32, 1.0001594642002871);  squeeze_32 = None
        mul_74 = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75 = torch.ops.aten.mul.Tensor(primals_83, 0.9);  primals_83 = None
        add_57 = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40 = torch.ops.aten.unsqueeze.default(primals_32, -1)
        unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42 = torch.ops.aten.unsqueeze.default(primals_33, -1);  primals_33 = None
        unsqueeze_43 = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_58 = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        relu_10 = torch.ops.aten.relu.default(add_58);  add_58 = None
        convolution_11 = torch.ops.aten.convolution.default(relu_10, primals_34, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_59 = torch.ops.aten.add.Tensor(primals_87, 1);  primals_87 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22 = var_mean_11[0]
        getitem_23 = var_mean_11[1];  var_mean_11 = None
        add_60 = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11 = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_11 = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77 = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33 = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34 = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78 = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79 = torch.ops.aten.mul.Tensor(primals_85, 0.9);  primals_85 = None
        add_61 = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35 = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_80 = torch.ops.aten.mul.Tensor(squeeze_35, 1.0001594642002871);  squeeze_35 = None
        mul_81 = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82 = torch.ops.aten.mul.Tensor(primals_86, 0.9);  primals_86 = None
        add_62 = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44 = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_45 = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46 = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_47 = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_63 = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        relu_11 = torch.ops.aten.relu.default(add_63);  add_63 = None
        convolution_12 = torch.ops.aten.convolution.default(relu_11, primals_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_64 = torch.ops.aten.add.Tensor(primals_90, 1);  primals_90 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24 = var_mean_12[0]
        getitem_25 = var_mean_12[1];  var_mean_12 = None
        add_65 = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12 = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
        sub_12 = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
        mul_84 = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36 = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37 = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85 = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86 = torch.ops.aten.mul.Tensor(primals_88, 0.9);  primals_88 = None
        add_66 = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38 = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_87 = torch.ops.aten.mul.Tensor(squeeze_38, 1.0001594642002871);  squeeze_38 = None
        mul_88 = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89 = torch.ops.aten.mul.Tensor(primals_89, 0.9);  primals_89 = None
        add_67 = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48 = torch.ops.aten.unsqueeze.default(primals_38, -1)
        unsqueeze_49 = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90 = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50 = torch.ops.aten.unsqueeze.default(primals_39, -1);  primals_39 = None
        unsqueeze_51 = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_68 = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        add_69 = torch.ops.aten.add.Tensor(add_68, relu_9);  add_68 = None
        relu_12 = torch.ops.aten.relu.default(add_69);  add_69 = None
        convolution_13 = torch.ops.aten.convolution.default(relu_12, primals_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_70 = torch.ops.aten.add.Tensor(primals_93, 1);  primals_93 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26 = var_mean_13[0]
        getitem_27 = var_mean_13[1];  var_mean_13 = None
        add_71 = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13 = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_13 = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91 = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39 = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_40 = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92 = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93 = torch.ops.aten.mul.Tensor(primals_91, 0.9);  primals_91 = None
        add_72 = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41 = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_94 = torch.ops.aten.mul.Tensor(squeeze_41, 1.0001594642002871);  squeeze_41 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96 = torch.ops.aten.mul.Tensor(primals_92, 0.9);  primals_92 = None
        add_73 = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52 = torch.ops.aten.unsqueeze.default(primals_41, -1)
        unsqueeze_53 = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97 = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54 = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
        unsqueeze_55 = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_74 = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        relu_13 = torch.ops.aten.relu.default(add_74);  add_74 = None
        convolution_14 = torch.ops.aten.convolution.default(relu_13, primals_43, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
        add_75 = torch.ops.aten.add.Tensor(primals_96, 1);  primals_96 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_14, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28 = var_mean_14[0]
        getitem_29 = var_mean_14[1];  var_mean_14 = None
        add_76 = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14 = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_14 = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98 = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42 = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43 = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99 = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100 = torch.ops.aten.mul.Tensor(primals_94, 0.9);  primals_94 = None
        add_77 = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44 = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_101 = torch.ops.aten.mul.Tensor(squeeze_44, 1.0001594642002871);  squeeze_44 = None
        mul_102 = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103 = torch.ops.aten.mul.Tensor(primals_95, 0.9);  primals_95 = None
        add_78 = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56 = torch.ops.aten.unsqueeze.default(primals_44, -1)
        unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104 = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58 = torch.ops.aten.unsqueeze.default(primals_45, -1);  primals_45 = None
        unsqueeze_59 = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_79 = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        relu_14 = torch.ops.aten.relu.default(add_79);  add_79 = None
        convolution_15 = torch.ops.aten.convolution.default(relu_14, primals_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_80 = torch.ops.aten.add.Tensor(primals_99, 1);  primals_99 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_15, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30 = var_mean_15[0]
        getitem_31 = var_mean_15[1];  var_mean_15 = None
        add_81 = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15 = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_15 = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
        mul_105 = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45 = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46 = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106 = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107 = torch.ops.aten.mul.Tensor(primals_97, 0.9);  primals_97 = None
        add_82 = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47 = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_108 = torch.ops.aten.mul.Tensor(squeeze_47, 1.0001594642002871);  squeeze_47 = None
        mul_109 = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110 = torch.ops.aten.mul.Tensor(primals_98, 0.9);  primals_98 = None
        add_83 = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60 = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111 = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62 = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
        unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_84 = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        add_85 = torch.ops.aten.add.Tensor(add_84, relu_12);  add_84 = None
        relu_15 = torch.ops.aten.relu.default(add_85);  add_85 = None
        convolution_16 = torch.ops.aten.convolution.default(relu_15, primals_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        add_86 = torch.ops.aten.add.Tensor(primals_102, 1);  primals_102 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32 = var_mean_16[0]
        getitem_33 = var_mean_16[1];  var_mean_16 = None
        add_87 = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16 = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_16 = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112 = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48 = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_49 = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113 = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114 = torch.ops.aten.mul.Tensor(primals_100, 0.9);  primals_100 = None
        add_88 = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50 = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_115 = torch.ops.aten.mul.Tensor(squeeze_50, 1.0001594642002871);  squeeze_50 = None
        mul_116 = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117 = torch.ops.aten.mul.Tensor(primals_101, 0.9);  primals_101 = None
        add_89 = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64 = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_65 = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118 = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66 = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_67 = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_90 = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        relu_16 = torch.ops.aten.relu.default(add_90);  add_90 = None
        le = torch.ops.aten.le.Scalar(relu_16, 0)
        unsqueeze_68 = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 2);  unsqueeze_68 = None
        unsqueeze_70 = torch.ops.aten.unsqueeze.default(unsqueeze_69, 3);  unsqueeze_69 = None
        unsqueeze_80 = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 2);  unsqueeze_80 = None
        unsqueeze_82 = torch.ops.aten.unsqueeze.default(unsqueeze_81, 3);  unsqueeze_81 = None
        unsqueeze_92 = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_93 = torch.ops.aten.unsqueeze.default(unsqueeze_92, 2);  unsqueeze_92 = None
        unsqueeze_94 = torch.ops.aten.unsqueeze.default(unsqueeze_93, 3);  unsqueeze_93 = None
        unsqueeze_104 = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_105 = torch.ops.aten.unsqueeze.default(unsqueeze_104, 2);  unsqueeze_104 = None
        unsqueeze_106 = torch.ops.aten.unsqueeze.default(unsqueeze_105, 3);  unsqueeze_105 = None
        unsqueeze_116 = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_117 = torch.ops.aten.unsqueeze.default(unsqueeze_116, 2);  unsqueeze_116 = None
        unsqueeze_118 = torch.ops.aten.unsqueeze.default(unsqueeze_117, 3);  unsqueeze_117 = None
        unsqueeze_128 = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_129 = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
        unsqueeze_130 = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
        unsqueeze_140 = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_141 = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
        unsqueeze_142 = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
        unsqueeze_152 = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_153 = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
        unsqueeze_154 = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
        unsqueeze_164 = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_165 = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
        unsqueeze_166 = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
        unsqueeze_176 = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_177 = torch.ops.aten.unsqueeze.default(unsqueeze_176, 2);  unsqueeze_176 = None
        unsqueeze_178 = torch.ops.aten.unsqueeze.default(unsqueeze_177, 3);  unsqueeze_177 = None
        unsqueeze_188 = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_189 = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
        unsqueeze_190 = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
        unsqueeze_200 = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_201 = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
        unsqueeze_202 = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
        unsqueeze_212 = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_213 = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214 = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
        unsqueeze_224 = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_225 = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
        unsqueeze_226 = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
        unsqueeze_236 = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_237 = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
        unsqueeze_238 = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
        unsqueeze_248 = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_249 = torch.ops.aten.unsqueeze.default(unsqueeze_248, 2);  unsqueeze_248 = None
        unsqueeze_250 = torch.ops.aten.unsqueeze.default(unsqueeze_249, 3);  unsqueeze_249 = None
        unsqueeze_260 = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_261 = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262 = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        return [add_2, add_3, add, add_8, add_9, add_6, add_13, add_14, add_11, add_18, add_19, add_16, add_24, add_25, add_22, add_29, add_30, add_27, add_34, add_35, add_32, add_40, add_41, add_38, add_45, add_46, add_43, add_50, add_51, add_48, add_56, add_57, add_54, add_61, add_62, add_59, add_66, add_67, add_64, add_72, add_73, add_70, add_77, add_78, add_75, add_82, add_83, add_80, add_88, add_89, add_86, relu, relu_16, relu_15, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_103, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, relu_2, convolution_3, squeeze_10, relu_3, convolution_4, squeeze_13, relu_4, convolution_5, squeeze_16, relu_5, convolution_6, squeeze_19, relu_6, convolution_7, squeeze_22, relu_7, convolution_8, squeeze_25, relu_8, convolution_9, squeeze_28, relu_9, convolution_10, squeeze_31, relu_10, convolution_11, squeeze_34, relu_11, convolution_12, squeeze_37, relu_12, convolution_13, squeeze_40, relu_13, convolution_14, squeeze_43, relu_14, convolution_15, squeeze_46, relu_15, convolution_16, squeeze_49, le, unsqueeze_70, unsqueeze_82, unsqueeze_94, unsqueeze_106, unsqueeze_118, unsqueeze_130, unsqueeze_142, unsqueeze_154, unsqueeze_166, unsqueeze_178, unsqueeze_190, unsqueeze_202, unsqueeze_214, unsqueeze_226, unsqueeze_238, unsqueeze_250, unsqueeze_262]
        
def load_args(reader):
    buf0 = reader.storage('81afdf8d038fff62d3199e09cbe66136cdb18906', 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1024, 512, 1, 1), requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf1, (1024,), requires_grad=True, is_leaf=True)  # primals_2
    buf2 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf2, (1024,), requires_grad=True, is_leaf=True)  # primals_3
    buf3 = reader.storage('a5c1aada06cfe106fda72e99c92e59033270231c', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf3, (256, 1024, 1, 1), requires_grad=True, is_leaf=True)  # primals_4
    buf4 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf4, (256,), requires_grad=True, is_leaf=True)  # primals_5
    buf5 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf5, (256,), requires_grad=True, is_leaf=True)  # primals_6
    buf6 = reader.storage('abe79e193aaa1bbd8d081502070603989e998181', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf6, (256, 256, 3, 3), requires_grad=True, is_leaf=True)  # primals_7
    buf7 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf7, (256,), requires_grad=True, is_leaf=True)  # primals_8
    buf8 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf8, (256,), requires_grad=True, is_leaf=True)  # primals_9
    buf9 = reader.storage('dde95e15989fda900e2dd0610525251c465a0449', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf9, (1024, 256, 1, 1), requires_grad=True, is_leaf=True)  # primals_10
    buf10 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf10, (1024,), requires_grad=True, is_leaf=True)  # primals_11
    buf11 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf11, (1024,), requires_grad=True, is_leaf=True)  # primals_12
    buf12 = reader.storage('d3bc140f4c307c412251a332205e6d34cf9cb39d', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf12, (256, 1024, 1, 1), requires_grad=True, is_leaf=True)  # primals_13
    buf13 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf13, (256,), requires_grad=True, is_leaf=True)  # primals_14
    buf14 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf14, (256,), requires_grad=True, is_leaf=True)  # primals_15
    buf15 = reader.storage('0584c59065c0088afc98a93d943f168df168d4ac', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf15, (256, 256, 3, 3), requires_grad=True, is_leaf=True)  # primals_16
    buf16 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf16, (256,), requires_grad=True, is_leaf=True)  # primals_17
    buf17 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf17, (256,), requires_grad=True, is_leaf=True)  # primals_18
    buf18 = reader.storage('8129ee06119062b1578c8d60416d1d935e49a6fc', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf18, (1024, 256, 1, 1), requires_grad=True, is_leaf=True)  # primals_19
    buf19 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf19, (1024,), requires_grad=True, is_leaf=True)  # primals_20
    buf20 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf20, (1024,), requires_grad=True, is_leaf=True)  # primals_21
    buf21 = reader.storage('c57f1b2963010c880e03f989555890801f8c7779', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf21, (256, 1024, 1, 1), requires_grad=True, is_leaf=True)  # primals_22
    buf22 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf22, (256,), requires_grad=True, is_leaf=True)  # primals_23
    buf23 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf23, (256,), requires_grad=True, is_leaf=True)  # primals_24
    buf24 = reader.storage('b8f70e43316c7edfc061016afcb168bedbe998c5', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf24, (256, 256, 3, 3), requires_grad=True, is_leaf=True)  # primals_25
    buf25 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf25, (256,), requires_grad=True, is_leaf=True)  # primals_26
    buf26 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf26, (256,), requires_grad=True, is_leaf=True)  # primals_27
    buf27 = reader.storage('4c7941954c8dc9eb64cd74e4a3ce6b42b27c722f', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf27, (1024, 256, 1, 1), requires_grad=True, is_leaf=True)  # primals_28
    buf28 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf28, (1024,), requires_grad=True, is_leaf=True)  # primals_29
    buf29 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf29, (1024,), requires_grad=True, is_leaf=True)  # primals_30
    buf30 = reader.storage('9938b00673bb8e43a66392ff283d23750f58eedf', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf30, (256, 1024, 1, 1), requires_grad=True, is_leaf=True)  # primals_31
    buf31 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf31, (256,), requires_grad=True, is_leaf=True)  # primals_32
    buf32 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf32, (256,), requires_grad=True, is_leaf=True)  # primals_33
    buf33 = reader.storage('27c045c0843a1364ac82c39292b5a255d64aabb8', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf33, (256, 256, 3, 3), requires_grad=True, is_leaf=True)  # primals_34
    buf34 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf34, (256,), requires_grad=True, is_leaf=True)  # primals_35
    buf35 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf35, (256,), requires_grad=True, is_leaf=True)  # primals_36
    buf36 = reader.storage('a6a1c868f6e731f18f3bd9ebd4f4cbe88d220a6e', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf36, (1024, 256, 1, 1), requires_grad=True, is_leaf=True)  # primals_37
    buf37 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf37, (1024,), requires_grad=True, is_leaf=True)  # primals_38
    buf38 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf38, (1024,), requires_grad=True, is_leaf=True)  # primals_39
    buf39 = reader.storage('b97820a8ec88bba1fb12e4b5f8ad4b571d24c404', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf39, (256, 1024, 1, 1), requires_grad=True, is_leaf=True)  # primals_40
    buf40 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf40, (256,), requires_grad=True, is_leaf=True)  # primals_41
    buf41 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf41, (256,), requires_grad=True, is_leaf=True)  # primals_42
    buf42 = reader.storage('f22bfe4c77dc3917019dd5aabe94e30b1c5858fb', 2359296, device=device(type='cuda', index=0))
    reader.tensor(buf42, (256, 256, 3, 3), requires_grad=True, is_leaf=True)  # primals_43
    buf43 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf43, (256,), requires_grad=True, is_leaf=True)  # primals_44
    buf44 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf44, (256,), requires_grad=True, is_leaf=True)  # primals_45
    buf45 = reader.storage('b9abc15e86801c75e82875e14df588c777704cbf', 1048576, device=device(type='cuda', index=0))
    reader.tensor(buf45, (1024, 256, 1, 1), requires_grad=True, is_leaf=True)  # primals_46
    buf46 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf46, (1024,), requires_grad=True, is_leaf=True)  # primals_47
    buf47 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf47, (1024,), requires_grad=True, is_leaf=True)  # primals_48
    buf48 = reader.storage('6a6a8f07d1da36c190b71f94883fe802092623d2', 2097152, device=device(type='cuda', index=0))
    reader.tensor(buf48, (512, 1024, 1, 1), requires_grad=True, is_leaf=True)  # primals_49
    buf49 = reader.storage('c4cd0520bfae1cd01ff7d836dd058cfc0644a4c9', 2048, device=device(type='cuda', index=0))
    reader.tensor(buf49, (512,), requires_grad=True, is_leaf=True)  # primals_50
    buf50 = reader.storage('41cd0520002e1cd04277d8363b058cfc34c4a4c9', 2048, device=device(type='cuda', index=0))
    reader.tensor(buf50, (512,), requires_grad=True, is_leaf=True)  # primals_51
    buf51 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf51, (1024,), is_leaf=True)  # primals_52
    buf52 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf52, (1024,), is_leaf=True)  # primals_53
    buf53 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf53, (), dtype=torch.int64, is_leaf=True)  # primals_54
    buf54 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf54, (256,), is_leaf=True)  # primals_55
    buf55 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf55, (256,), is_leaf=True)  # primals_56
    buf56 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf56, (), dtype=torch.int64, is_leaf=True)  # primals_57
    buf57 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf57, (256,), is_leaf=True)  # primals_58
    buf58 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf58, (256,), is_leaf=True)  # primals_59
    buf59 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf59, (), dtype=torch.int64, is_leaf=True)  # primals_60
    buf60 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf60, (1024,), is_leaf=True)  # primals_61
    buf61 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf61, (1024,), is_leaf=True)  # primals_62
    buf62 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf62, (), dtype=torch.int64, is_leaf=True)  # primals_63
    buf63 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf63, (256,), is_leaf=True)  # primals_64
    buf64 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf64, (256,), is_leaf=True)  # primals_65
    buf65 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf65, (), dtype=torch.int64, is_leaf=True)  # primals_66
    buf66 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf66, (256,), is_leaf=True)  # primals_67
    buf67 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf67, (256,), is_leaf=True)  # primals_68
    buf68 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf68, (), dtype=torch.int64, is_leaf=True)  # primals_69
    buf69 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf69, (1024,), is_leaf=True)  # primals_70
    buf70 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf70, (1024,), is_leaf=True)  # primals_71
    buf71 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf71, (), dtype=torch.int64, is_leaf=True)  # primals_72
    buf72 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf72, (256,), is_leaf=True)  # primals_73
    buf73 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf73, (256,), is_leaf=True)  # primals_74
    buf74 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf74, (), dtype=torch.int64, is_leaf=True)  # primals_75
    buf75 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf75, (256,), is_leaf=True)  # primals_76
    buf76 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf76, (256,), is_leaf=True)  # primals_77
    buf77 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf77, (), dtype=torch.int64, is_leaf=True)  # primals_78
    buf78 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf78, (1024,), is_leaf=True)  # primals_79
    buf79 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf79, (1024,), is_leaf=True)  # primals_80
    buf80 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf80, (), dtype=torch.int64, is_leaf=True)  # primals_81
    buf81 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf81, (256,), is_leaf=True)  # primals_82
    buf82 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf82, (256,), is_leaf=True)  # primals_83
    buf83 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf83, (), dtype=torch.int64, is_leaf=True)  # primals_84
    buf84 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf84, (256,), is_leaf=True)  # primals_85
    buf85 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf85, (256,), is_leaf=True)  # primals_86
    buf86 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf86, (), dtype=torch.int64, is_leaf=True)  # primals_87
    buf87 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf87, (1024,), is_leaf=True)  # primals_88
    buf88 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf88, (1024,), is_leaf=True)  # primals_89
    buf89 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf89, (), dtype=torch.int64, is_leaf=True)  # primals_90
    buf90 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf90, (256,), is_leaf=True)  # primals_91
    buf91 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf91, (256,), is_leaf=True)  # primals_92
    buf92 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf92, (), dtype=torch.int64, is_leaf=True)  # primals_93
    buf93 = reader.storage('46eaa2fd283e76f1285df65b0bfa53ba2201b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf93, (256,), is_leaf=True)  # primals_94
    buf94 = reader.storage('416aa2fd47be76f169ddf65bc17a53bab801b60a', 1024, device=device(type='cuda', index=0))
    reader.tensor(buf94, (256,), is_leaf=True)  # primals_95
    buf95 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf95, (), dtype=torch.int64, is_leaf=True)  # primals_96
    buf96 = reader.storage('6924efa03a0749155601a6f85c2eb9651b489f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf96, (1024,), is_leaf=True)  # primals_97
    buf97 = reader.storage('19a4efa0300749158e01a6f8952eb965a9c89f6c', 4096, device=device(type='cuda', index=0))
    reader.tensor(buf97, (1024,), is_leaf=True)  # primals_98
    buf98 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf98, (), dtype=torch.int64, is_leaf=True)  # primals_99
    buf99 = reader.storage('41cd0520002e1cd04277d8363b058cfc34c4a4c9', 2048, device=device(type='cuda', index=0))
    reader.tensor(buf99, (512,), is_leaf=True)  # primals_100
    buf100 = reader.storage('c4cd0520bfae1cd01ff7d836dd058cfc0644a4c9', 2048, device=device(type='cuda', index=0))
    reader.tensor(buf100, (512,), is_leaf=True)  # primals_101
    buf101 = reader.storage('6de79f4d451761873a72c46a249b835d43d5f5e8', 8, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf101, (), dtype=torch.int64, is_leaf=True)  # primals_102
    buf102 = reader.storage('a8c0441a4abb80932abe9676d37d61de2667ce30', 51380224, device=device(type='cuda', index=0))
    reader.tensor(buf102, (32, 512, 28, 28), requires_grad=True)  # primals_103
    buf103 = reader.storage('dc67357d3eeb916b24095287bad18381867e25d4', 25690112, device=device(type='cuda', index=0))
    reader.tensor(buf103, (32, 1024, 14, 14), requires_grad=True)  # primals_104
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=True, command='run', save_dir='/public/moco/torch_compile_debug/run_2024_03_04_03_01_41_905565-pid_70786/minifier/checkpoints', tracing_mode='real', check_str=None)

# Checking accuracy of non-floating point values such as boolean tensors
# can lead to false positives.