import torch
import torch.nn as nn
import torch.nn.functional as F
import math

[optim]
optimizer = "adam"
lr = 0.001
weight_decay = 1.0e-5

class DCRN(nn.Module):
    def __init__(self, rnn_hidden=128, fft_len=512, kernel_size=5, kernel_num=[16, 32, 64, 128, 128, 128]):
        super(DCRN, self).__init__()
        self.rnn_hidden = rnn_hidden
        self.fft_len = fft_len

        self.kernel_size = kernel_size
        self.kernel_num = [2,] + kernel_num

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for idx in range(len(self.kernel_num) - 1):
            self.encoder.append(
                CausalConv(                 
                        self.kernel_num[idx],
                        self.kernel_num[idx + 1],
                        kernel_size=(self.kernel_size, 2),
                        stride=(2, 1)
                        )
            )
        hidden_dim = self.fft_len // (2 ** (len(self.kernel_num)))

        self.enhance = nn.LSTM(
            input_size=hidden_dim * self.kernel_num[-1],
            hidden_size=self.rnn_hidden,
            num_layers=1,
            dropout=0.0,
            batch_first=False
        )
        self.transform = nn.Linear(self.rnn_hidden, hidden_dim * self.kernel_num[-1])
        for idx in range(len(self.kernel_num) - 1, 0, -1):
            if idx != 1:
                self.decoder.append(
                    CausalTransConv(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                    )
                )
            else:
                self.decoder.append(
                        CausalTransConvEnd(
                            self.kernel_num[idx] * 2,
                            self.kernel_num[idx - 1],
                            kernel_size=(self.kernel_size, 2),
                            stride=(2, 1),
                            padding=(2, 0),
                            output_padding=(1, 0)
                        )
                )
        if isinstance(self.enhance, nn.LSTM):
            self.enhance.flatten_parameters()    

    def forward(self, real, imag, spec_mags):
        spec_complex = torch.stack([real, imag], dim=1)[:, :, 1:]
        out = self.compress(spec_mags[:, 1:, :], spec_complex)
        encoder_out = []
        
        for idx, encoder in enumerate(self.encoder):
            out = encoder(out)
            encoder_out.append(out)

        B, C, D, T = out.shape
        out = out.permute(3, 0, 1, 2)
        out = out.reshape(T, B, C * D)
        out, _ = self.enhance(out)
        out = self.transform(out)
        out = out.reshape(T, B, C, D)
        out = out.permute(1, 2, 3, 0)

        for idx in range(len(self.decoder)):
            out = torch.cat([out, encoder_out[-1 - idx]], 1)
            out = self.decoder[idx](out)

        mask_real = out[:, 0]
        mask_imag = out[:, 1]
        mask_real = F.pad(mask_real, [0, 0, 1, 0], value=1e-8)
        mask_imag = F.pad(mask_imag, [0, 0, 1, 0], value=1e-8) 

        return mask_real, mask_imag

    def compress(self, mag, spec_complex):
        scaler = torch.unsqueeze(mag ** 0.23 / (mag + 1e-8), 1) 
        spec_complex = spec_complex * scaler 
        return spec_complex

    def denoisy(self, noisy, win_len = 512,hop_len = 128,fft_len = 512):
        stft  = ConvSTFT(win_len, hop_len, fft_len, 'hann', 'complex')
        istft = ConviSTFT(win_len,hop_len, fft_len, 'hann', 'complex')
        noisy_spec = stft(noisy)
        real = noisy_spec[:, :fft_len // 2 + 1]
        imag = noisy_spec[:, fft_len // 2 + 1:]
        spec_mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        spec_phase = torch.atan(imag / (real + 1e-8))
        phase_adjust = (real < 0).to(torch.int) * torch.sign(imag) * math.pi
        spec_phase = spec_phase + phase_adjust
        mask_real, mask_imag = self(real, imag, spec_mags)
        mask_mags = (mask_real ** 2 + mask_imag ** 2) ** 0.5
        real_phase = mask_real / (mask_mags + 1e-8)
        imag_phase = mask_imag / (mask_mags + 1e-8)
        mask_phase = torch.atan(imag_phase / (real_phase + 1e-8))
        phase_adjust = (real_phase < 0).to(torch.int) * torch.sign(imag_phase) * math.pi
        mask_phase = mask_phase + phase_adjust
        mask_mags = torch.tanh(mask_mags)  
        est_mags = mask_mags * spec_mags
        est_phase = spec_phase + mask_phase
        real = est_mags * torch.cos(est_phase)
        imag = est_mags * torch.sin(est_phase)
        est_spec = torch.cat([real, imag], 1)
        est_wav = istft(est_spec).squeeze(1)
        return  est_wav

def si_snr(s1, s2, eps=1e-8):
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target = s1_s2_norm / (s2_s2_norm + eps) * s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)