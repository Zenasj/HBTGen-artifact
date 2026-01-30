model_to_quantize_fx=quantize_fx.prepare_qat_fx(model_to_quantize,qconfig_dict)        
model_to_quantize_fx.apply(torch.quantization.disable_observer)
torch.onnx.export(model_to_quantize_fx,torch.randn(1,1,60,60).cuda(),'xxxxx.onnx',opset_version=13)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization.quantize_fx as quantize_fx
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3)
        self.conv_1 = nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(2,4), stride=(2,1), padding=2, dilation=1)
    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        return x

def test():
    net = Model()
    net.train()
    qconfig_dict={"":torch.quantization.get_default_qat_qconfig('fbgemm')}
    net=quantize_fx.prepare_qat_fx(net,qconfig_dict)
    x = torch.rand(1, 12, 64, 64)
    result=net(x)
    net.apply(torch.quantization.disable_observer)
    torch.onnx.export(net,x,'xxxxx3.onnx',opset_version=13)
if __name__ == "__main__":
    test()