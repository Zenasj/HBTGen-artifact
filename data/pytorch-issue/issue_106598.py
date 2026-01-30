import torch.nn as nn

import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv_test=torch.nn.Conv2d(1,3,(3,3))
    def forward(self,x):
        x=self.conv_test(x)
        return x
input=torch.randn(1,1,15,15)
model=Model()

import torch.ao.quantization.quantize_fx as quantize_fx
import copy

model_to_quantize=copy.deepcopy(model)
model_to_quantize.train()

from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.qconfig import default_symmetric_qnnpack_qat_qconfig

qconfig=default_symmetric_qnnpack_qat_qconfig
qconfig_mapping = QConfigMapping().set_global(qconfig)
backend_config=torch.ao.quantization.backend_config.qnnpack.get_qnnpack_backend_config()
torch.backends.quantized.engine = 'qnnpack'

model_prepared = quantize_fx.prepare_qat_fx(model_to_quantize,qconfig_mapping=qconfig_mapping,example_inputs=input,backend_config=backend_config)
model_prepared(input)
model_prepared(input)
model_quantized = quantize_fx.convert_fx(model_prepared,qconfig_mapping=qconfig_mapping,backend_config=backend_config)


model_trace=torch.jit.trace(model_quantized,input)
torch.jit.save(model_trace,'model_one_conv.pth')

model_trace_reload=torch.jit.load('model_one_conv.pth')

result_ref=model_quantized(input)
result_model_trace_reload=model_trace_reload(input)
equal_flag=torch.allclose(result_ref,result_model_trace_reload)
print('Result equal flag:{}'.format(equal_flag))
if equal_flag:
    print('The qnnpack config jit model can be executed on my x86_64 machine')

NoQEngine
ONEDNN
X86
FBGEMM