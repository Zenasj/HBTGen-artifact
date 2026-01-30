import torch
import torchaudio

theta = torch.linspace(0, 300 * 3.14, 8000)
waveform = torch.sin(theta).unsqueeze(0)
print(waveform.shape)
n_fft = 400
window = torch.hann_window(n_fft)
for center in [True, False]:
    print(center)
    spec = torch.stft(waveform, n_fft=n_fft, center=center, window=window, return_complex=True)
    print('spec:', spec.shape, spec.dtype)
    recon = torch.istft(spec, n_fft=n_fft, center=center, window=window, length=8000) # note length here
    torchaudio.save(f'recon_center_{center}.wav', recon, 8000)
    print('recon:', recon.shape)
    torch.testing.assert_allclose(waveform, recon)

approx_nonzero_indices = ifft_window_sum > util.tiny(ifft_window_sum)
y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

py
x = torch.tensor(x) # any signal
nfft = 128
w = torch.hann_window(nfft+2)[1:-1]
s = torch.stft(x, n_fft=nfft, window=w, center=False, return_complex=True)
x_hat = torch.istft(s, n_fft=nfft, window=w, center=False)

py
print(torch.hann_window(16).numpy())
print(scipy.signal.windows.hann(16))