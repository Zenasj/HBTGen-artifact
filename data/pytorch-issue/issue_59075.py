import torchvision

import torch, torchvision
import torch.quantization as quant
qconfig = quant.QConfig(
    activation=quant.FakeQuantize.with_args(
        observer=quant.fake_quantize.HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False),
    weight=quant.FakeQuantize.with_args(
        observer=quant.fake_quantize.MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False))
model = torchvision.models.quantization.mobilenet_v2()
model.qconfig = qconfig
quant.prepare_qat(model, inplace=True)
model.cuda()
model(torch.rand(12, 3, 224, 224).cuda())

import torch, torchvision
import torch.quantization as quant

model = torchvision.models.quantization.mobilenet_v2()
model.qconfig = quant.get_default_qconfig("qnnpack")
model.qconfig = qconfig
print(model.qconfig)
quant.prepare_qat(model, inplace=True)

model.cuda()
model(torch.rand(12, 3, 224, 224).cuda())