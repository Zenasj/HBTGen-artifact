import torch
import torch.nn as nn

x = torch.zeros(1, 12, 80)
torch.backends.quantized.engine = 'qnnpack'
model = torch.nn.LSTM(80, 48, 4, batch_first=True, bidirectional=True, dropout=0.0)
quant_model = torch.quantization.quantize_dynamic(
                model, {nn.LSTM}, dtype=torch.qint8
)
torch.set_num_threads(1)
traced = torch.jit.trace(quant_model, x)