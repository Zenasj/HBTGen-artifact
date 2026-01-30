# built from source: https://github.com/descriptinc/descript-audio-codec/tree/main
import dac
import torch
from audiotools import AudioSignal

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to('cuda')

# Load audio signal file
# 4ch.wav is located here: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples/SoundCardAttrition/4ch.wav
signal = AudioSignal('4ch.wav')
signal._audio_data = signal._audio_data[:, :1, :]
# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)

@torch.compile
def _compress(signal):
    return model.compress(signal)
x = _compress(signal)

# Decompress it back to an AudioSignal
@torch.compile
def _decompress(x):
    return model.decompress(x)
y = _decompress(x)