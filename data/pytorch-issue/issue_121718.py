import logging
import torch
import torchaudio
torch._logging.set_logs(dynamo = logging.DEBUG)
torch._dynamo.config.verbose = True,

layer = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=128, window_fn=torch.hann_window).to('cuda')
layer_compiled = torch.compile(layer)

x = torch.randn(1, 1, 3000).to('cuda') # (bsz, 1, frame)
y = layer_compiled(x) # <--- error here!