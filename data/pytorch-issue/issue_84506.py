import torch

a, b = torch.tensor([0.5, 0.5]), torch.tensor([0.33, 0.33])

def foo(x, y):
    nt = torch.nested_tensor([x, y])
    return nt * nt
z = make_fx(foo)(a, b)

def forward(self, x_1, y_1):
    nested_tensor = torch.ops.aten.nested_tensor.default([x_1, y_1]);  x_1 = y_1 = None
    getitem = nested_tensor[0]
    getitem_1 = nested_tensor[1];  nested_tensor = None
    _tensor_constant0 = self._tensor_constant0
    _tensor_constant0_1 = self._tensor_constant0
    mul = torch.ops.aten.mul.Tensor(_tensor_constant0, _tensor_constant0_1);  _tensor_constant0 = _tensor_constant0_1 = None
    getitem_2 = mul[0]
    getitem_3 = mul[1];  mul = None
    _tensor_constant1 = self._tensor_constant1
    return _tensor_constant1

def forward(self, x_1, y_1):
    nested_tensor = torch.ops.aten.nested_tensor.default([x_1, y_1]);  x_1 = y_1 = None
    getitem = nested_tensor[0]
    getitem_1 = nested_tensor[1];  nested_tensor = None
    select = torch.ops.aten.select.int(getitem_1, 0, 0);  getitem_1 = None
    return select

def forward(self, x_1, y_1):
    nested_tensor = torch.ops.aten.nested_tensor.default([x_1, y_1]);  x_1 = y_1 = None
    getitem = nested_tensor[0]
    getitem_1 = nested_tensor[1];  nested_tensor = None
    _tensor_constant0 = self._tensor_constant0
    select = torch.ops.aten.select.int(_tensor_constant0, 0, 0);  _tensor_constant0 = None
    return select