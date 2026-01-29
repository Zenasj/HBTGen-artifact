# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (batch_size, seq_length)
import torch
import torchaudio
from torch import nn

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_batches=1, num_biquads=1, num_layers=1, fs=44100):
        super(MyModel, self).__init__()
        self.eps = 1e-8
        self.fs = fs
        self.dirac = self.get_dirac(fs, 0, grad=True)  # generate a dirac
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, output_size)
        )
        self.sos = torch.rand(num_biquads, 6, device='cpu', dtype=torch.float32, requires_grad=True)

    def get_dirac(self, size, index=1, grad=False):
        tensor = torch.zeros(size, requires_grad=grad)
        tensor.data[index] = 1
        return tensor

    def compute_filter_magnitude_and_phase_frequency_response(self, dirac, fs, a, b):
        filtered_dirac = torchaudio.functional.lfilter(dirac, a, b)
        freqs_response = torch.fft.fft(filtered_dirac)
        freqs_rad = torch.fft.rfftfreq(filtered_dirac.shape[-1])
        freqs_hz = freqs_rad[:filtered_dirac.shape[-1] // 2] * fs / np.pi
        freqs_response = freqs_response[:len(freqs_hz)]
        mag_response_db = 20 * torch.log10(torch.abs(freqs_response))
        phase_response_rad = torch.angle(freqs_response)
        phase_response_deg = phase_response_rad * 180 / np.pi
        return freqs_hz, mag_response_db, phase_response_deg

    def forward(self, x):
        self.sos = self.mlp(x)
        return self.sos

def my_model_function():
    return MyModel(seq_length, 10*seq_length, 6, batche_size, num_biquads, 1, fs)

def GetInput():
    # Generate a random input tensor with the same shape as the training input
    input_seq_batch = torch.rand(batche_size, seq_length, dtype=torch.float32)
    return input_seq_batch

# Constants
fs = 1024
num_biquads = 1
seq_length = 512
batche_size = 1

