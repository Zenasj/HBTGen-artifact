import torch.nn as nn

import torch.onnx
import torchaudio
from torch import nn


class DataCov(nn.Module):
    def __init__(self):
        super(DataCov, self).__init__()

        self.transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_fft=1536, hop_length=768, f_min=20, f_max=20000)
        )

    def forward(self, x1):
        return self.transform(x1)



if __name__ == '__main__':
    model = DataCov()
    model.eval()
    model = torch.jit.script(model)

    x = torch.randn(1, 48000 * 12, requires_grad=True)
    args = (x,)
    torch.onnx.export(model, args, 'DataCov.onnx', export_params=True, opset_version=15)