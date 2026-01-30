import torch.nn as nn

import torch

class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x, h):
        new_h = torch.tanh(self.linear(x) + h)
        return new_h, new_h

my_cell = MyCell()
x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))

#find aten::add node
graph = traced_cell.graph.nodes()
node_iter = next(graph)
while node_iter.kind() != 'aten::add':
    node_iter = next(graph)

#set debug_name attribute in jit graph
node_iter.s_('debug_name','MyCell/add1')

print(traced_cell)
#graph(%self.1 : __torch__.MyCell,
#      %x : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu),
#      %h : Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)):
#  %linear : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
#  %20 : Tensor = prim::CallMethod[name="forward"](%linear, %x)
#  %11 : int = prim::Constant[value=1]() # /home/lbachar/pytorch/test.py:9:0
#  %12 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::add[debug_name="MyCell/add1"](%20, %h, %11) # /home/lbachar/pytorch/test.py:9:0
#  %13 : Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu) = aten::tanh(%12) # /home/lbachar/pytorch/test.py:9:0
#  %14 : (Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu), Float(3, 4, strides=[4, 1], requires_grad=1, device=cpu)) = prim::TupleConstruct(%13, %13)
#  return (%14)