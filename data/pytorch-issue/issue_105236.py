import torch

def fn(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out.to_local()

def forward(self, primals_1, primals_2):
    view = torch.ops.aten.view.default(primals_1, [2, 4]);  primals_1 = None
    _to_copy = torch.ops.aten._to_copy.default(view, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  view = None
    detach = torch.ops.aten.detach.default(_to_copy);  _to_copy = None
    detach_1 = torch.ops.aten.detach.default(detach);  detach = None
    view_1 = torch.ops.aten.view.default(primals_2, [4, 2]);  primals_2 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(view_1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  view_1 = None
    detach_2 = torch.ops.aten.detach.default(_to_copy_1);  _to_copy_1 = None
    detach_3 = torch.ops.aten.detach.default(detach_2);  detach_2 = None
    detach_4 = torch.ops.aten.detach.default(detach_1)
    all_gather_into_tensor = torch.ops.c10d_functional.all_gather_into_tensor.default(detach_3, 'ptd:0', [0, 1], 2)
    wait_tensor = torch.ops.c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
    split = torch.ops.aten.split.Tensor(wait_tensor, 4);  wait_tensor = None
    getitem = split[0]
    getitem_1 = split[1];  split = None
    cat = torch.ops.aten.cat.default([getitem, getitem_1], 1);  getitem = getitem_1 = None
    detach_5 = torch.ops.aten.detach.default(cat);  cat = None
    mm = torch.ops.aten.mm.default(detach_4, detach_5);  detach_4 = detach_5 = None
    detach_6 = torch.ops.aten.detach.default(mm);  mm = None
    detach_9 = torch.ops.aten.detach.default(detach_6);  detach_6 = None
    detach_10 = torch.ops.aten.detach.default(detach_9);  detach_9 = None
    t = torch.ops.aten.t.default(detach_1);  detach_1 = None
    detach_13 = torch.ops.aten.detach.default(t);  t = None
    t_1 = torch.ops.aten.t.default(detach_3);  detach_3 = None
    detach_15 = torch.ops.aten.detach.default(t_1);  t_1 = None
    clone = torch.ops.aten.clone.default(detach_15, memory_format = torch.contiguous_format);  detach_15 = None
    return [detach_10, detach_13, clone]

def forward(self, detach_13, clone, tangents_1):
    detach_11 = torch.ops.aten.detach.default(tangents_1);  tangents_1 = None
    detach_12 = torch.ops.aten.detach.default(detach_11);  detach_11 = None
    mm_1 = torch.ops.aten.mm.default(detach_13, detach_12);  detach_13 = None
    detach_14 = torch.ops.aten.detach.default(mm_1);  mm_1 = None
    detach_16 = torch.ops.aten.detach.default(detach_12);  detach_12 = None
    all_gather_into_tensor_2 = torch.ops.c10d_functional.all_gather_into_tensor.default(clone, 'ptd:0', [0, 1], 2);  clone = None
    wait_tensor_2 = torch.ops.c10d_functional.wait_tensor.default(all_gather_into_tensor_2);
    detach_17 = torch.ops.aten.detach.default(wait_tensor_2);  wait_tensor_2 = None
    mm_2 = torch.ops.aten.mm.default(detach_16, detach_17);  detach_16 = detach_17 = None
    detach_18 = torch.ops.aten.detach.default(mm_2);  mm_2 = None
    split_1 = torch.ops.aten.split.Tensor(detach_14, 2, 1);  detach_14 = None
    getitem_2 = split_1[0]
    getitem_3 = split_1[1];  split_1 = None
    cat_1 = torch.ops.aten.cat.default([getitem_2, getitem_3]);  getitem_2 = getitem_3 = None
    reduce_scatter_tensor = torch.ops.c10d_functional.reduce_scatter_tensor.default(cat_1, 'SUM', 'ptd:0', [0, 1], 2);  cat_1 = None
    wait_tensor_3 = torch.ops.c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
    detach_19 = torch.ops.aten.detach.default(wait_tensor_3);  wait_tensor_3 = None
    detach_20 = torch.ops.aten.detach.default(detach_19);  detach_19 = None
    detach_21 = torch.ops.aten.detach.default(detach_20);  detach_20 = None
    detach_22 = torch.ops.aten.detach.default(detach_21);  detach_21 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(detach_22, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  detach_22 = None
    view_2 = torch.ops.aten.view.default(_to_copy_2, [8]);  _to_copy_2 = None
    detach_23 = torch.ops.aten.detach.default(detach_18);  detach_18 = None
    detach_24 = torch.ops.aten.detach.default(detach_23);  detach_23 = None
    _to_copy_3 = torch.ops.aten._to_copy.default(detach_24, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'));  detach_24 = None
    view_3 = torch.ops.aten.view.default(_to_copy_3, [8]);  _to_copy_3 = None
    return [view_3, view_2]