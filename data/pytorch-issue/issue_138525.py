import torch.nn as nn

import torch


class STFTModel(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 1024,
        n_hop: int = 256,
        input_channels: int = 2,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.conv = torch.nn.Conv2d(
            in_channels=input_channels * 2,
            out_channels=input_channels * 2,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        )

    def forward(self, x):
        b, c, t = x.size()
        x = x.reshape(-1, t)
        stft_output = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            center=True,
            onesided=True,
            pad_mode="reflect",
            return_complex=True,
        )
        _, f, frames = stft_output.shape
        stft_output = stft_output.reshape(b, c, f, frames).permute(0, 1, 3, 2)
        out = torch.stack([torch.real(stft_output), torch.imag(stft_output)], dim=-1)
        out = torch.permute(out, (0, 1, 4, 2, 3))
        out = torch.reshape(out, (-1, c * 2, frames, f)).contiguous()
        out = self.conv(out)
        t2 = out.shape[2]
        out = torch.reshape(out, (b, c, 2, t2, f))
        out = torch.permute(out, (0, 1, 4, 3, 2))
        out = torch.view_as_complex(out.contiguous())
        out = out.reshape(-1, f, t2)
        out = torch.istft(
            out,
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            center=True,
            onesided=True,
            length=t,
        )
        return out.reshape(b, c, t)


device = torch.device("mps")
model = STFTModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
num_samples = 100
batch_size = 16
input_length = 16000

for _ in range(5):
    for _ in range(num_samples // batch_size):
        x = torch.randn(batch_size, 2, input_length).to(device)
        target = torch.randn(batch_size, 2, input_length).to(device)
        output = model(x)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()