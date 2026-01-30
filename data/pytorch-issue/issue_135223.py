import torch

device = torch.device('cpu')
freq_cpu = torch.fft.fftfreq(10**4, device=device).to('mps').cpu()
device = torch.device('mps')
freq_mps = torch.fft.fftfreq(10**4, device=device).cpu()

print('CPU: fftfreq ranges from %f to %f' % (freq_cpu.min(), freq_cpu.max()) )
print('MPS: fftfreq ranges from %f to %f' % (freq_mps.min(), freq_mps.max()) )