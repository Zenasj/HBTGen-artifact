import torch

ViewInfo=(x.base(), x.size(), x.stride, x,storage_offset())

auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default,
     _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0 ,
     _y_base_index = 0,_y_size = (), _y_stride = (), _y_storage_offset = 1   ,
     _all_bases = [arg0_1])

def forward(self, arg0_1: "f32[2][1]cpu"):
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops.mylib.foo.default, _x_base_index = 0, _x_size = (), _x_stride = (), _x_storage_offset = 0, _y_base_index = 0, _y_size = (), _y_stride = (), _y_storage_offset = 1, _all_bases = [arg0_1])
        getitem_1: "f32[2][1]cpu" = auto_functionalized[1];  auto_functionalized = None
        copy_: "f32[2][1]cpu" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = copy_ = None
        
        # No stacktrace found for following nodes
        select_2: "f32[][]cpu" = torch.ops.aten.select.int(getitem_1, 0, 0)
        select_3: "f32[][]cpu" = torch.ops.aten.select.int(getitem_1, 0, 1);  getitem_1 = None
        return (select_2, select_3)