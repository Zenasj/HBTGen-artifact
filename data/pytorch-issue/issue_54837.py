import torch
import torch.nn as nn
class M_Instance_Norm(torch.nn.Module):
    def __init__(self):
        super(M_Instance_Norm, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.norm = nn.InstanceNorm3d(32, affine=True)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.norm(x)
        x = self.dequant(x)
        return x

def test_quant(model_fp32, input):
    model_fp32.eval()

    res_fp32 = model_fp32(input)

    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.quantization.prepare(model_fp32)

    model_fp32_prepared(input)

    model_int8 = torch.quantization.convert(model_fp32_prepared)

    # run the model, relevant calculations will happen in int8
    res_int8 = model_int8(input)
    print(model_fp32)
    print(model_int8)

    print(torch.allclose(res_int8, res_fp32, 1.0, 1e-01))

test_quant(M_Instance_Norm(), torch.randn(1, 32, 224, 224, 160))

test_quant(M_Instance_Norm(), torch.randn(1, 32, 224, 224, 160)) #fail
test_quant(M_Instance_Norm(), torch.randn(1, 32, 224, 224, 16)) #pass