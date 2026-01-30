import torch.nn as nn
import torch.nn.functional as F

#### MODEL ####
class aslTinyModel(nn.Module):
    def __init__(self):
        super(aslTinyModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(INPUT_LEN, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, NUM_GESTURES)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = self.dequant(h)
        return h

model = aslTinyModel()

#### TRAIN ####

#### QUANTIZE ####
import torch.quantization
num_calibration_batches = 10
model.eval()

model.qconfig = torch.quantization.default_qconfig
qconfig = torch.quantization.get_default_qconfig('qnnpack')
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

#### SAVE TO ONNX ####
dummy_input = torch.randn(1, INPUT_LEN)
torch.onnx.export(model, (dummy_input), "./asl_qmodel.onnx", verbose=True, opset_version=10)