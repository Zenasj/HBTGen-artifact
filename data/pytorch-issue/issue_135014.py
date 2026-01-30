import torch.nn as nn

import torch


class istft_class(torch.nn.Module):
    def __init__(self):
        super(istft_class, self).__init__()

        self.window = torch.hann_window(1024).type(torch.FloatTensor)

    def forward(self, spec):

        output = torch.istft(
            spec,
            n_fft=1024,
            hop_length=512,
            window=self.window,
            length=144000,
        )

        return output


def istft_dynamo_export_demo():
    model = istft_class()
    model.eval()

    real_part = torch.randn(1, 513, 282, dtype=torch.float32)
    imaginary_part = torch.randn(1, 513, 282, dtype=torch.float32)
    spec = torch.complex(real_part, imaginary_part)

    onnx_program = torch.onnx.dynamo_export(model, spec)
    onnx_program.save("./data/istft_model_dynamo.onnx")


if __name__ == "__main__":
    istft_dynamo_export_demo()

import torch


class istft_class(torch.nn.Module):
    def __init__(self):
        super(istft_class, self).__init__()

        self.window = torch.hann_window(1024).type(torch.FloatTensor)

    def forward(self, spec):

        output = torch.istft(
            spec,
            n_fft=1024,
            hop_length=512,
            window=self.window,
            length=144000,
        )

        return output



model = istft_class()
model.eval()

real_part = torch.randn(1, 513, 282, dtype=torch.float32)
imaginary_part = torch.randn(1, 513, 282, dtype=torch.float32)
spec = torch.complex(real_part, imaginary_part)

ep = torch.export.export(model, (spec,))
print(ep)