py
import torch
import torch.nn as nn
import torch.nn.functional as F
import ai_edge_torch

class PadLayer(nn.Module):
    def __init__(self):
        super(PadLayer, self).__init__()
        self.paddings = (0, 0, 1, 1)  # Padding for the second and third dimensions

    def forward(self, x):
        return F.pad(x, pad=self.paddings, mode="constant", value=-1)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 7), padding='same')
        self.pad = PadLayer()
        self.mpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=0)

    def forward(self, x):
        x = self.conv(x)
        x = self.pad(x)
        x = self.mpool(x)
        return x

model = SimpleModel()

example_input = torch.randn(1, 1, 32, 25)

edge_model = ai_edge_torch.convert(model.eval(), (example_input,))

# Export the model to TFLite format
edge_model.export('simple_model_with_pad.tflite')