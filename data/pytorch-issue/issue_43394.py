import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Sequential(
                nn.ConstantPad2d(padding*2, 0),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride)
            )
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        return self.dequant(self.conv(x))

in_channels, out_channels, kernel_size, stride, padding = 3, 5, 3, 1, (1,2)
model = Model(3, 5, 3, 1, (1,2))
model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)

# Calibrate first
evaluate(model, criterion, data_loader, neval_batches=num_calibration_batches)

# Convert to quantized model
torch.quantization.convert(model, inplace=True)

# Evaluation (Quantized) 
top1, top5 = evaluate(model, criterion, data_loader_test, neval_batches=num_eval_batches) #This line gives error.