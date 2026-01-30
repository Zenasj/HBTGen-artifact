3
import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

class TEST(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.convt = torch.nn.ConvTranspose2d(3, 3, 3)

    def forward(self, x):
        x = self.quant(x)
        x = self.convt(x)
        x = self.dequant(x)
        return x

rand_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    model = TEST().eval()
    backend = "qnnpack" #"fbgemm" # this var also error
    torch.backends.quantized.engine = backend
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    script_module = torch.jit.script(model)
    optimized_model = optimize_for_mobile(script_module, backend='metal')
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter("quant_test.ptl")