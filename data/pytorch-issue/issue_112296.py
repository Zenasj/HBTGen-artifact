import torch.nn as nn
import torch.nn.functional as F

import torch
from torch._dynamo import compiled_autograd


def fn_simple(x, y):
    a = torch.cos(x)
    b = torch.sin(y)
    c = a + b
    loss = c - 0.1
    loss.backward(torch.tensor(3.0))
    return loss

x = torch.tensor(100.0, requires_grad=True)
y = torch.tensor(200.0, requires_grad=True)

fn_simple_dynamo = torch.compile(fn_simple, backend="eager")
loss = fn_simple_dynamo(x, y)

cls._wrapped_call = _WrappedCall(cls, cls_call)  # type: ignore[attr-defined]

if isinstance(mod, torch.fx.GraphModule):
                    # TODO: do we want to support __call__ for GM's?
                    # If so at least some changes are needed, we don't allow inlining
                    # the call_wrapped currently, and maybe other issues too
                    fn = mod.forward

def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    getitem_2 = L_x_
    getitem_1 = L_y_
    a = torch.cos(getitem_2)
    b = torch.sin(getitem_1)
    c = a + b;  a = b = None
    loss = c - 0.1;  c = None
    getitem = torch.tensor(3.0)
    cos_1 = torch.ops.aten.cos.default(getitem_1);  getitem_1 = None
    mul = torch.ops.aten.mul.Tensor(getitem, cos_1);  cos_1 = None
    new_empty_strided = torch.ops.aten.new_empty_strided.default(mul, [], [], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    copy_ = torch.ops.aten.copy_.default(new_empty_strided, mul);  new_empty_strided = mul = None
    sin_1 = torch.ops.aten.sin.default(getitem_2);  getitem_2 = None
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(getitem, neg);  getitem = neg = None
    new_empty_strided_1 = torch.ops.aten.new_empty_strided.default(mul_1, [], [], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    copy__1 = torch.ops.aten.copy_.default(new_empty_strided_1, mul_1);  new_empty_strided_1 = mul_1 = None
    return (loss,)

def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    getitem_2 = L_x_
    getitem_1 = L_y_
    a = torch.cos(getitem_2)
    b = torch.sin(getitem_1)
    c = a + b;  a = b = None
    loss = c - 0.1;  c = None
    getitem = torch.tensor(3.0)
    cos_1 = torch.ops.aten.cos.default(getitem_1)
    mul = torch.ops.aten.mul.Tensor(getitem, cos_1);  cos_1 = None
    accumulate_grad__default = torch.ops.inductor.accumulate_grad_.default(getitem_1, mul);  getitem_1 = mul = None
    sin_1 = torch.ops.aten.sin.default(getitem_2)
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(getitem, neg);  getitem = neg = None
    accumulate_grad__default_1 = torch.ops.inductor.accumulate_grad_.default(getitem_2, mul_1);  getitem_2 = mul_1 = None
    return (loss,)

t1.grad = grad1
t2.grad = grad2
...

with compiled_autograd.enable(dummy_fn_to_save_fx):
                self.proxy.node.meta.get("example_value").backward()

def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):
    getitem_4 = L_x_
    getitem_3 = L_y_
    
    # File: xla/test.py:6, code: a = torch.cos(x)
    a = torch.cos(getitem_4)
    
    # File: xla/test.py:7, code: b = torch.sin(y)
    b = torch.sin(getitem_3)
    
    # File: xla/test.py:8, code: c = a + b
    c = a + b;  a = b = None
    
    # File: xla/test.py:9, code: loss = c - 0.1
    loss = c - 0.1;  c = None
    
    # File: xla/test.py:10, code: loss.backward(torch.tensor(5.0, device=a.device))
    tensor = torch.tensor(5.0, device = device(type='cpu'))
    
    # File: /src/pytorch/torch/autograd/__init__.py:246, code: grad_tensors_ = _tensor_or_tensors_to_tuple(grad_tensors, len(tensors))
    _tensor_or_tensors_to_tuple = torch.autograd._tensor_or_tensors_to_tuple(tensor, 1);  tensor = None
    getitem = _tensor_or_tensors_to_tuple[0];  _tensor_or_tensors_to_tuple = None
    
    # File: /src/pytorch/torch/autograd/__init__.py:247, code: grad_tensors_ = _make_grads(tensors, grad_tensors_, is_grads_batched=False)
    _make_grads = torch.autograd._make_grads((loss,), (getitem,), is_grads_batched = False);  getitem = None
    getitem_2 = _make_grads[0];  _make_grads = None
    
    # File: <eval_with_key>.0:8, code: cos = torch.ops.aten.cos.default(getitem_1)
    cos_1 = torch.ops.aten.cos.default(getitem_3);  getitem_3 = None
    
    # File: <eval_with_key>.0:9, code: mul = torch.ops.aten.mul.Tensor(getitem, cos);  cos = None
    mul = torch.ops.aten.mul.Tensor(getitem_2, cos_1);  cos_1 = None
    
    # File: <eval_with_key>.0:11, code: sin = torch.ops.aten.sin.default(getitem_2)
    sin_1 = torch.ops.aten.sin.default(getitem_4);  getitem_4 = None
    
    # File: <eval_with_key>.0:12, code: neg = torch.ops.aten.neg.default(sin);  sin = None
    neg = torch.ops.aten.neg.default(sin_1);  sin_1 = None
    
    # File: <eval_with_key>.0:13, code: mul_1 = torch.ops.aten.mul.Tensor(getitem, neg);  getitem = neg = None
    mul_1 = torch.ops.aten.mul.Tensor(getitem_2, neg);  getitem_2 = neg = None
    return (loss, mul_1, mul)

_make_grads = torch.autograd._make_grads((loss,), (getitem,), is_grads_batched = False);

class XlaMNIST(nn.Module):

  def __init__(self):
    super(XlaMNIST, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2(x), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

def train_model(model, data, target):
  loss_fn = torch.nn.CrossEntropyLoss()
  pred = model(data)
  loss = loss_fn(pred, target)
  #loss.backward()
  return pred

def forward(self, L_data_ : torch.Tensor, L_target_ : torch.Tensor):
    l_data_ = L_data_
    l_target_ = L_target_
    
    # File: xla/test_mnist.py:19, code: x = F.relu(F.max_pool2d(self.conv1(x), 2))
    l__model___conv1 = self.L__model___conv1(l_data_);  l_data_ = None
    max_pool2d = torch.nn.functional.max_pool2d(l__model___conv1, 2);  l__model___conv1 = None
    x = torch.nn.functional.relu(max_pool2d);  max_pool2d = None
    
    # File: xla/test_mnist.py:20, code: x = F.relu(F.max_pool2d(self.conv2(x), 2))
    l__model___conv2 = self.L__model___conv2(x);  x = None
    max_pool2d_1 = torch.nn.functional.max_pool2d(l__model___conv2, 2);  l__model___conv2 = None
    x_1 = torch.nn.functional.relu(max_pool2d_1);  max_pool2d_1 = None
    
    # File: xla/test_mnist.py:21, code: x = x.view(-1, 320)
    x_2 = x_1.view(-1, 320);  x_1 = None
    
    # File: xla/test_mnist.py:22, code: x = F.relu(self.fc1(x))
    l__model___fc1 = self.L__model___fc1(x_2);  x_2 = None
    x_3 = torch.nn.functional.relu(l__model___fc1);  l__model___fc1 = None
    
    # File: xla/test_mnist.py:23, code: x = self.fc2(x)
    x_4 = self.L__model___fc2(x_3);  x_3 = None
    
    # File: xla/test_mnist.py:24, code: return F.log_softmax(x, dim=1)
    pred = torch.nn.functional.log_softmax(x_4, dim = 1);  x_4 = None
    
    # File: xla/test_mnist.py:29, code: loss = loss_fn(pred, target)
    loss = torch.nn.functional.cross_entropy(pred, l_target_, None, None, -100, None, 'mean', 0.0);  l_target_ = None
    return (pred,)

def forward(self, L_data_ : torch.Tensor, L_target_ : torch.Tensor):
    l_data_ = L_data_
    l_target_ = L_target_
    
    # File: /src/pytorch/torch/nn/modules/conv.py:460, code: return self._conv_forward(input, self.weight, self.bias)
    l__model___conv1_weight = self.L__model___conv1_weight
    l__model___conv1_bias = self.L__model___conv1_bias
    
    # File: /src/pytorch/torch/nn/modules/conv.py:456, code: return F.conv2d(input, weight, bias, self.stride,
    conv2d = torch.conv2d(l_data_, l__model___conv1_weight, l__model___conv1_bias, (1, 1), (0, 0), (1, 1), 1);  l_data_ = l__model___conv1_weight = l__model___conv1_bias = None
    
    # File: xla/test_mnist.py:19, code: x = F.relu(F.max_pool2d(self.conv1(x), 2))
    max_pool2d = torch.nn.functional.max_pool2d(conv2d, 2);  conv2d = None
    x = torch.nn.functional.relu(max_pool2d);  max_pool2d = None
    
    # File: /src/pytorch/torch/nn/modules/conv.py:460, code: return self._conv_forward(input, self.weight, self.bias)
    l__model___conv2_weight = self.L__model___conv2_weight
    l__model___conv2_bias = self.L__model___conv2_bias
    
    # File: /src/pytorch/torch/nn/modules/conv.py:456, code: return F.conv2d(input, weight, bias, self.stride,
    conv2d_1 = torch.conv2d(x, l__model___conv2_weight, l__model___conv2_bias, (1, 1), (0, 0), (1, 1), 1);  x = l__model___conv2_weight = l__model___conv2_bias = None
    
    # File: xla/test_mnist.py:20, code: x = F.relu(F.max_pool2d(self.conv2(x), 2))
    max_pool2d_1 = torch.nn.functional.max_pool2d(conv2d_1, 2);  conv2d_1 = None
    x_1 = torch.nn.functional.relu(max_pool2d_1);  max_pool2d_1 = None
    
    # File: xla/test_mnist.py:21, code: x = x.view(-1, 320)
    x_2 = x_1.view(-1, 320);  x_1 = None
    
    # File: /src/pytorch/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
    l__model___fc1_weight = self.L__model___fc1_weight
    l__model___fc1_bias = self.L__model___fc1_bias
    linear = torch._C._nn.linear(x_2, l__model___fc1_weight, l__model___fc1_bias);  x_2 = l__model___fc1_weight = l__model___fc1_bias = None
    
    # File: xla/test_mnist.py:22, code: x = F.relu(self.fc1(x))
    x_3 = torch.nn.functional.relu(linear);  linear = None
    
    # File: /src/pytorch/torch/nn/modules/linear.py:114, code: return F.linear(input, self.weight, self.bias)
    l__model___fc2_weight = self.L__model___fc2_weight
    l__model___fc2_bias = self.L__model___fc2_bias
    x_4 = torch._C._nn.linear(x_3, l__model___fc2_weight, l__model___fc2_bias);  x_3 = l__model___fc2_weight = l__model___fc2_bias = None
    
    # File: xla/test_mnist.py:24, code: return F.log_softmax(x, dim=1)
    pred = torch.nn.functional.log_softmax(x_4, dim = 1);  x_4 = None
    
    # File: xla/test_mnist.py:29, code: loss = loss_fn(pred, target)
    loss = torch.nn.functional.cross_entropy(pred, l_target_, None, None, -100, None, 'mean', 0.0);  l_target_ = None
    return (pred,)