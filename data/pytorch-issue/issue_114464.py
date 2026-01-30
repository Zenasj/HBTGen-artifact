import torch

def eager(x):
    y = x.clone().reshape(-1, 4)
    y[:, [2, 0]] = y[:, [0, 2]]
    return y


compiled = torch.compile(backend="inductor", fullgraph=True)(eager)

x = torch.tensor([0, 1, 2, 3])

print(f"{eager(x)=}")
print(f"{x=}, {x._version=}, {id(x)=}, {x.data_ptr()=}")
print(f"{compiled(x)=}")
print(f"{x=}, {x._version=}, {id(x)=}, {x.data_ptr()=}")

import torch

def eager(x):
    y = x.clone().reshape(-1, 4)
    y[:, [2, 0]] = y[:, [0, 2]]
    return y


compiled = torch.compile(backend="inductor", fullgraph=True)(eager)

x = torch.tensor([0, 1, 2, 3])

print(f"{x.data_ptr()=}")
print(f"{eager(x).data_ptr()=}")
print(f"{compiled(x).data_ptr()=}")

def forward(self, arg0_1: "i64[4]"):
    # File: <ipython-input-1-74b5d2352691>:4, code: y = x.clone().reshape(-1, 4)
    clone: "i64[4]" = torch.ops.aten.clone.default(arg0_1);  arg0_1 = None
    view: "i64[1, 4]" = torch.ops.aten.view.default(clone, [-1, 4])
    
    # File: <ipython-input-1-74b5d2352691>:5, code: y[:, [2, 0]] = y[:, [0, 2]]
    slice_1: "i64[1, 4]" = torch.ops.aten.slice.Tensor(view, 0, 0, 9223372036854775807)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "i64[2]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    index: "i64[1, 2]" = torch.ops.aten.index.Tensor(slice_1, [None, lift_fresh_copy]);  slice_1 = lift_fresh_copy = None
    slice_2: "i64[1, 4]" = torch.ops.aten.slice.Tensor(view, 0, 0, 9223372036854775807);  view = None
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "i64[2]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    view_1: "i64[2]" = torch.ops.aten.view.default(index, [2]);  index = None
    index_put: "i64[1, 4]" = torch.ops.aten.index_put.default(slice_2, [None, lift_fresh_copy_1], view_1);  slice_2 = lift_fresh_copy_1 = view_1 = None
    view_2: "i64[1, 4]" = torch.ops.aten.view.default(clone, [-1, 4]);  clone = None
    slice_scatter: "i64[1, 4]" = torch.ops.aten.slice_scatter.default(view_2, index_put, 0, 0, 9223372036854775807);  view_2 = index_put = None
    view_3: "i64[4]" = torch.ops.aten.view.default(slice_scatter, [4]);  slice_scatter = None
    
    # No stacktrace found for following nodes
    view_5: "i64[1, 4]" = torch.ops.aten.view.default(view_3, [-1, 4]);  view_3 = None
    return (view_5,)

def forward(self, arg0_1: "i64[4]"):
    # File: <ipython-input-1-74b5d2352691>:5, code: y[:, [2, 0]] = y[:, [0, 2]]
    _tensor_constant0: "i64[2]" = self._tensor_constant0
    _tensor_constant1: "i64[2]" = self._tensor_constant1
    view_2: "i64[1, 4]" = torch.ops.aten.reshape.default(arg0_1, [-1, 4])
      
    # File: <ipython-input-1-74b5d2352691>:4, code: y = x.clone().reshape(-1, 4)
    view: "i64[1, 4]" = torch.ops.aten.reshape.default(arg0_1, [-1, 4]);  arg0_1 = None
      
    # File: <ipython-input-1-74b5d2352691>:5, code: y[:, [2, 0]] = y[:, [0, 2]]
    lift_fresh_copy_1: "i64[2]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    lift_fresh_copy: "i64[2]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    index: "i64[1, 2]" = torch.ops.aten.index.Tensor(view, [None, lift_fresh_copy]);  lift_fresh_copy = None
    view_1: "i64[2]" = torch.ops.aten.reshape.default(index, [2]);  index = None
    index_put: "i64[1, 4]" = torch.ops.aten.index_put.default(view, [None, lift_fresh_copy_1], view_1);  view = lift_fresh_copy_1 = view_1 = None
    view_3: "i64[4]" = torch.ops.aten.reshape.default(index_put, [4]);  index_put = None
      
    # No stacktrace found for following nodes
    view_5: "i64[1, 4]" = torch.ops.aten.reshape.default(view_3, [-1, 4]);  view_3 = None
    return (view_5,)

def forward(self, arg0_1: "i64[4]"):
    # File: <ipython-input-1-74b5d2352691>:5, code: y[:, [2, 0]] = y[:, [0, 2]]
    _tensor_constant0: "i64[2]" = self._tensor_constant0
    _tensor_constant1: "i64[2]" = self._tensor_constant1
    view_2: "i64[1, 4]" = torch.ops.aten.reshape.default(arg0_1, [-1, 4])
    
    # File: <ipython-input-1-74b5d2352691>:4, code: y = x.clone().reshape(-1, 4)
    view: "i64[1, 4]" = torch.ops.aten.reshape.default(arg0_1, [-1, 4]);  arg0_1 = None
     
    # File: <ipython-input-1-74b5d2352691>:5, code: y[:, [2, 0]] = y[:, [0, 2]]
    lift_fresh_copy_1: "i64[2]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    lift_fresh_copy: "i64[2]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    index: "i64[1, 2]" = torch.ops.aten.index.Tensor(view, [None, lift_fresh_copy]);  lift_fresh_copy = None
    view_1: "i64[2]" = torch.ops.aten.reshape.default(index, [2]);  index = None
    index_put: "i64[1, 4]" = torch.ops.aten.index_put_.default(view, [None, lift_fresh_copy_1], view_1);  view = lift_fresh_copy_1 = view_1 = None
    view_3: "i64[4]" = torch.ops.aten.reshape.default(index_put, [4]);  index_put = None
    
    # No stacktrace found for following nodes
    view_5: "i64[1, 4]" = torch.ops.aten.reshape.default(view_3, [-1, 4]);  view_3 = None
    return (view_5,)