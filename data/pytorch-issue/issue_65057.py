import random

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
import torch.onnx.symbolic_helper as sym_help
from torch.autograd.function import once_differentiable
import json
from torch.nn.modules.utils import _pair
from torch.onnx.symbolic_helper import parse_args

class Quantizer(Function):
    @staticmethod
    def symbolic(g,input):
        return g.op('Qconv',input)
    @staticmethod
    def forward(ctx, input):
        return torch.round(input * 3) / 3

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output,None

def quantize(input):
    return Quantizer.apply(input)
    
class TestM(nn.Module):
    def __init__(self):
        super(TestM,self).__init__()
        self.conv1=nn.Conv2d(3,64,3)
        self.quant=quantize

    def forward(self,input):
        output=self.conv1(input)
        output=self.quant(output)
        return output

if __name__=='__main__':

    model=TestM()
    input=torch.ones(1,3,224,224)
   
    torch.onnx.export(
        model,
        input,
        'test.onnx',
        input_names=["input"],
        output_names=['output'],
        verbose=True
    )

import math
import time
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import torch.nn.functional as F
import torch.onnx.symbolic_helper as sym_help
from torch.autograd.function import once_differentiable
import json
from torch.nn.modules.utils import _pair
from torch.onnx.symbolic_helper import parse_args
from torch.onnx import register_custom_op_symbolic
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


class Quantizer(Function):
    @staticmethod
    def symbolic(g,input):
        return g.op("ai.onnx.contrib::Quantizer",input)
    @staticmethod
    def forward(ctx, input):
        return torch.round(input * 3) / 3

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output,None

# @onnx_op(op_type='Quantizer', domain='ai.onnx.contrib',inputs=[PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float])
# def Quantizer(q):
#     return q

def quantize(input):
    return Quantizer.apply(input)
    
class TestM(nn.Module):
    def __init__(self):
        super(TestM,self).__init__()
        self.conv1=nn.Conv2d(3,64,3)
        self.quant=quantize

    def forward(self,input):
        output=self.conv1(input)
        output=self.quant(output)
        return output

if __name__=='__main__':

    model=TestM()
    input=torch.ones(1,3,224,224)
    torch.onnx.export(
         model,
         input,
        'test.onnx',
        verbose=True
    )

@onnx_op(op_type='Quantizer', domain='ai.onnx.contrib',inputs=[PyCustomOpDef.dt_float], outputs=[PyCustomOpDef.dt_float])
def Quantizer(q):
    return q
if __name__=='__main__':

    # model=TestM()
    # input=torch.ones(1,3,224,224)
    # torch.onnx.export(
    #      model,
    #      input,
    #     'test.onnx',
    #     verbose=True
    # )
    import onnxruntime as rt
    import numpy as  np
    import onnx
    so = rt.SessionOptions()
    so.register_custom_ops_library(_get_library_path())
    onnx_model = onnx.load("test.onnx")
    onnx.checker.check_model(onnx_model)
    data = np.array(np.random.randn(1,3,224,224))
    sess = rt.InferenceSession('test.onnx',so)
    
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    pred_onx = sess.run([label_name], {input_name:data.astype(np.float32)})[0]
    print(pred_onx)