import torch

def forward(self, inputs, sizes, hooks):
    getitem = inputs[0]
    getitem_1 = inputs[1]
    getitem_2 = inputs[2]
    getitem_3 = inputs[3]
    getitem_4 = inputs[4]
    getitem_5 = inputs[5]
    getitem_6 = inputs[6]
    getitem_7 = inputs[7]
    getitem_8 = inputs[8]
    getitem_9 = inputs[9];  inputs = None
    expand = torch.ops.aten.expand.default(getitem, [2, 4]);  getitem = None
    threshold_backward = torch.ops.aten.threshold_backward.default(expand, getitem_1, 0);  expand = getitem_1 = None
    t = torch.ops.aten.t.default(getitem_3);  getitem_3 = None
    mm = torch.ops.aten.mm.default(threshold_backward, t);  t = None
    t_1 = torch.ops.aten.t.default(threshold_backward)
    mm_1 = torch.ops.aten.mm.default(t_1, getitem_2);  t_1 = getitem_2 = None
    t_2 = torch.ops.aten.t.default(mm_1);  mm_1 = None
    sum_1 = torch.ops.aten.sum.dim_IntList(threshold_backward, [0], True);  threshold_backward = None
    view = torch.ops.aten.view.default(sum_1, [4]);  sum_1 = None
    t_3 = torch.ops.aten.t.default(t_2);  t_2 = None
    accumulate_grad_ = torch.ops.inductor.accumulate_grad_.default(getitem_4, t_3);  getitem_4 = t_3 = None
    threshold_backward_1 = torch.ops.aten.threshold_backward.default(mm, getitem_5, 0);  mm = getitem_5 = None
    t_4 = torch.ops.aten.t.default(threshold_backward_1)
    mm_2 = torch.ops.aten.mm.default(t_4, getitem_6);  t_4 = getitem_6 = None
    t_5 = torch.ops.aten.t.default(mm_2);  mm_2 = None
    sum_2 = torch.ops.aten.sum.dim_IntList(threshold_backward_1, [0], True);  threshold_backward_1 = None
    view_1 = torch.ops.aten.view.default(sum_2, [4]);  sum_2 = None
    t_6 = torch.ops.aten.t.default(t_5);  t_5 = None
    accumulate_grad__1 = torch.ops.inductor.accumulate_grad_.default(getitem_7, t_6);  getitem_7 = t_6 = None
    accumulate_grad__2 = torch.ops.inductor.accumulate_grad_.default(getitem_8, view_1);  getitem_8 = view_1 = None
    accumulate_grad__3 = torch.ops.inductor.accumulate_grad_.default(getitem_9, view);  getitem_9 = view = None
    return []