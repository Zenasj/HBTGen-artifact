import torch.nn as nn

import torch
from torch import nn
import torchaudio

class DataCov(nn.Module):
    def __init__(self):
        super(DataCov, self).__init__()

        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1536, hop_length=768, f_min=20, f_max=20000)
        )

    def forward(self, x1):
        return self.transform(x1)


model = DataCov()
input = torch.rand((1, 1, 12 * 48000))
torch.onnx.export(model, (input), "DataCov.onnx", verbose=False,
                      input_names=['input1'], output_names=['output1'], opset_version=18)