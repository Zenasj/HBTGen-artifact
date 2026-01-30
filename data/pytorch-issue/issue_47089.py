import torch
import torch.nn as nn
import math

class BoxCoder(object):
    def __init__(self, bbox_xform_clip):
        # type: (float) -> None
        self.bbox_xform_clip = bbox_xform_clip

    def decode(self, rel_codes, boxes):
        # type: (Tensor, List[Tensor]) -> Tensor
        boxes_per_image = [b.size(0) for b in boxes]
        boxes = torch.cat(boxes, dim=0)

        pred_ctr_x = torch.clamp(rel_codes[:, 0::4], max=self.bbox_xform_clip)* boxes[:, 2]
        pred_ctr_y = torch.clamp(rel_codes[:, 1::4], max=self.bbox_xform_clip)* boxes[:, 3]

        return torch.stack((pred_ctr_x, pred_ctr_y), dim=2)

def test_1():
    class MyModule(torch.nn.Module):
        __annotations__ = {
            'box_coder': BoxCoder,
        }

        def __init__(self):
            super(MyModule, self).__init__()
            self.box_coder = BoxCoder(math.log(1000. / 16))

        def forward(self, box_regression, proposals):
            return self.box_coder.decode(box_regression, proposals)

    model = MyModule()
    model.eval()
    box_regression = torch.randn([4, 4])
    proposal = [torch.randn(2, 4), torch.randn(2, 4)]

    output = model(box_regression, proposal)
    script_m = torch.jit.script(model)

    script_m.eval()
    f_s_m = torch._C._freeze_module(script_m._c)
    f_s_m.dump()
    
# module __torch__.___torch_mangle_0.MyModule {
#   parameters {
#   }
#   attributes {
#     box_coder = <__torch__.BoxCoder object at 0x557d68413ae0>
#   }
#   methods {
#     method forward {
#       graph(%self : __torch__.___torch_mangle_0.MyModule,
#             %box_regression.1 : Tensor,
#             %proposals.1 : Tensor):
#         %12 : None = prim::Constant()
#         %11 : int = prim::Constant[value=9223372036854775807]()
#         %10 : int = prim::Constant[value=0]() # pytorch_test.py:13:34
#         %9 : int = prim::Constant[value=4]() # pytorch_test.py:16:49
#         %8 : int = prim::Constant[value=2]() # pytorch_test.py:16:89
#         %7 : int = prim::Constant[value=1]() # pytorch_test.py:17:46
#         %6 : int = prim::Constant[value=3]() # pytorch_test.py:17:89
#         %3 : __torch__.BoxCoder = prim::GetAttr[name="box_coder"](%self)
#         %4 : Tensor[] = prim::ListConstruct(%proposals.1)
#         %boxes.4 : Tensor = aten::cat(%4, %10) # pytorch_test.py:14:16
#         %14 : Tensor = aten::slice(%box_regression.1, %10, %10, %11, %7) # pytorch_test.py:16:33
#         %15 : Tensor = aten::slice(%14, %7, %10, %11, %9) # pytorch_test.py:16:33
#         %16 : float = prim::GetAttr[name="bbox_xform_clip"](%3)
#         %17 : Tensor = aten::clamp(%15, %12, %16) # pytorch_test.py:16:21
#         %18 : Tensor = aten::slice(%boxes.4, %10, %10, %11, %7) # pytorch_test.py:16:80
#         %19 : Tensor = aten::select(%18, %7, %8) # pytorch_test.py:16:80
#         %pred_ctr_x.1 : Tensor = aten::mul(%17, %19) # pytorch_test.py:16:21
#         %22 : Tensor = aten::slice(%14, %7, %7, %11, %9) # pytorch_test.py:17:33
#         %24 : Tensor = aten::clamp(%22, %12, %16) # pytorch_test.py:17:21
#         %26 : Tensor = aten::select(%18, %7, %6) # pytorch_test.py:17:80
#         %pred_ctr_y.1 : Tensor = aten::mul(%24, %26) # pytorch_test.py:17:21
#         %28 : Tensor[] = prim::ListConstruct(%pred_ctr_x.1, %pred_ctr_y.1)
#         %29 : Tensor = aten::stack(%28, %8) # pytorch_test.py:19:15
#         return (%29)

#     }
#   }
#   submodules {
#   }
# }