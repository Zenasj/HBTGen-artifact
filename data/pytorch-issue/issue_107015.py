import torch.nn as nn

import torch
import torchaudio
from torch import nn
from torch import onnx


class DataCov(nn.Module):
    def __init__(self):
        super(DataCov, self).__init__()

        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1536, hop_length=768, f_min=20, f_max=20000)
        )

    def forward(self, x1):
        return self.transform(x1)

def export_datacov_onnx(path):
    model = DataCov()
    model.eval()
    model = torch.jit.script(model)
    x = torch.randn((1, 48000 * 12), requires_grad=True)
    args = (x,)
    torch.onnx.dynamo_export(model, args, path, export_params=True, opset_version=17)

if __name__ == '__main__':
    export_datacov_onnx('DataCov.onnx')