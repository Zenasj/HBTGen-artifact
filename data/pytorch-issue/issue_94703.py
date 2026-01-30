import torch
import torch.nn as nn
from torch.autograd import Function
import torch.onnx

#custom op onnx representation
class MyStrangeOp(Function):
   #for onnx node representation, symbolic function must be defined and specified static.
   @staticmethod
   def symbolic(g, input, weight, bias, floatAttr, intAttr):
      #because forward function return 2 outputs, so this func also have to return 2 outputs
      #this is my expriment result, I didn't find any docs, fuck it!
      return g.op("MyStrangeOp", input, weight, bias, float_attr_f=floatAttr, int_attr_i=intAttr), \
             g.op("MyStrangeOp", input, weight, bias, float_attr_f=floatAttr, int_attr_i=intAttr)

   @staticmethod
   def forward(ctx, input, weight, bias, floatAttr, intAttr):
      #this op return 2 outputs
      return input + weight, input * weight + bias

myStrangeOpForward = MyStrangeOp.apply

#layer
class MyStrangeOpLayer(nn.Module):
   def __init__(self, weight, bias, floatAttr, intAttr):
      super(MyStrangeOpLayer, self).__init__()
      self.weight = weight
      self.bias = bias
      self.floatAttr = floatAttr
      self.intAttr = intAttr

   def forward(self, x):
      return myStrangeOpForward(x, self.weight, self.bias, self.floatAttr, self.intAttr)

#model
class MyStrangeNet(nn.Module):
   def __init__(self):
      super(MyStrangeNet, self).__init__()
      self.myLayer1 = MyStrangeOpLayer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), bias=nn.Parameter(torch.ones(1, 3, 4, 4)), floatAttr=[0.1, 0.5], intAttr=[2, 2])
      self.myLayer2 = MyStrangeOpLayer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), bias=nn.Parameter(torch.ones(1, 3, 4, 4)), floatAttr=[0.5, 0.5], intAttr=[3, 3])
      self.conv1    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
      self.conv2    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
      self.conv3    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
      self.conv4    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)

   def forward(self, x):
      x1, x2 = self.myLayer1(x)
      x3, x4 = self.myLayer2(x)
      x1     = self.conv1(x1)
      x2     = self.conv1(x2)
      x3     = self.conv1(x3)
      x4     = self.conv1(x4)
      return x1 + x2 + x3 + x4



model = MyStrangeNet()
t = torch.ones(1, 3, 4, 4, dtype=torch.float32)
torch.onnx.export(model, (t,), 'fuckIt.onnx', opset_version=13, input_names=["inputTensor"], output_names=["outputTensor"],
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

torch.onnx.register_custom_op_symbolic("prim::PythonOp", mycast_op, 11)