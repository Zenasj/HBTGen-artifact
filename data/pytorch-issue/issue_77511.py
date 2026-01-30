3
import torch
import torch.nn as nn


if __name__ == '__main__':
    model = nn.Sequential(
        torch.quantization.QuantStub(),
        nn.Conv1d(40, 40, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.AvgPool1d(32, ceil_mode=True),
        torch.quantization.DeQuantStub()
    )
    model.eval()
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    quant_prepared = torch.quantization.prepare(model)
    input_data = torch.randn((32, 40, 32))
    quant_prepared(input_data)
    quant_int8 = torch.quantization.convert(quant_prepared)
    input_data = torch.randn((32, 40, 16))
    print(quant_int8.forward(input_data).shape)