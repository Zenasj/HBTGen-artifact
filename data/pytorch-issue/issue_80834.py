import torch.nn as nn

import torch

class FeaturizeTorchFFT(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.stft(
            input=x,
            n_fft=320,
            hop_length=160,
            win_length=320,
            window=torch.ones(320),
            center=False,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )

    def export(self):
        with torch.no_grad():
            feature_inp = torch.arange(0, 1000).reshape(1, -1).float()
            output = self.forward(feature_inp)
            print('output shape ',output.shape)

        torch.onnx.export(
            self,  # model being run
            feature_inp,  # model input (or a tuple for multiple inputs)
            str('fft.onnx'),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["x"],
            output_names=["output"],
            dynamic_axes={
                "x": {0: "batch_size", 1: "audio_length"},
            },
        )

if __name__ == '__main__':
    converter = FeaturizeTorchFFT()
    converter.export()