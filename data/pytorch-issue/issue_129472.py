import torch.nn as nn
import random

import time
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from torchaudio.functional import lfilter
from torch.optim import Adam, lr_scheduler

# Set the device
hardware = "cpu"
device = torch.device(hardware)

class FilterNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_batches=1, num_biquads=1, num_layers=1, fs=44100):
        super(FilterNet, self).__init__()
        self.eps = 1e-8
        self.fs = fs
        self.dirac = self.get_dirac(fs, 0, grad=True)  # generate a dirac
        self.mlp = torch.nn.Sequential(torch.nn.Linear(input_size, 100),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(100, 50),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(50, output_size))
        self.sos = torch.rand(num_biquads, 6, device=hardware, dtype=torch.float32, requires_grad=True)
        

    def get_dirac(self, size, index=1, grad=False):
        tensor = torch.zeros(size, requires_grad=grad)
        tensor.data[index] = 1
        return tensor

    def compute_filter_magnitude_and_phase_frequency_response(self, dirac, fs, a, b):
        # filter it 
        filtered_dirac = lfilter(dirac, a, b) 
        freqs_response = torch.fft.fft(filtered_dirac)
        
        # compute the frequency axis (positive frequencies only)
        freqs_rad = torch.fft.rfftfreq(filtered_dirac.shape[-1])
        
        # keep only the positive freqs
        freqs_hz = freqs_rad[:filtered_dirac.shape[-1] // 2] * fs / np.pi
        freqs_response = freqs_response[:len(freqs_hz)]
        
        # magnitude response 
        mag_response_db = 20 * torch.log10(torch.abs(freqs_response))
        
        # Phase Response
        phase_response_rad = torch.angle(freqs_response)
        phase_response_deg = phase_response_rad * 180 / np.pi
        return freqs_hz, mag_response_db, phase_response_deg
        
    def forward(self, x):
        self.sos = self.mlp(x)
        return self.sos


# Define the target filter variables
fs = 2048 # 44100             # Sampling frequency
num_biquads = 1        # Number of biquad filters in the cascade
num_biquad_coeffs = 6  # Number of coefficients per biquad
   
# define filter coeffs
target_sos = torch.tensor([0.803, -0.132, 0.731, 1.000, -0.426, 0.850])
a = target_sos[3:]
b = target_sos[:3]

# prepare data
import scipy.signal as signal 
f0 = 20
f1 = 20e3
t = np.linspace(0, 60, fs, dtype=np.float32)
sine_sweep   = signal.chirp(t=t, f0=f0, t1=60, f1=f1, method='logarithmic')
white_noise  = np.random.normal(scale=5e-2, size=len(t)) 
noisy_sweep  = sine_sweep + white_noise
train_input  = torch.from_numpy(noisy_sweep.astype(np.float32))
train_target = lfilter(train_input, a, b) 

# Init the optimizer 
n_epochs    = 9
batche_size = 1
seq_length  = 512
seq_step    = 512
model     = FilterNet(seq_length, 10*seq_length, 6, batche_size, num_biquads, 1, fs)
optimizer = Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = torch.nn.MSELoss()

# compute filter response
freqs_hz, mag_response_db, phase_response_deg = model.compute_filter_magnitude_and_phase_frequency_response(model.get_dirac(fs, 0, grad=False), fs, a, b)
target_frequency_response = torch.hstack((mag_response_db, phase_response_deg))

# Inits
start_time = time.time()    # Start timing the loop
pbar = tqdm(total=n_epochs) # Create a tqdm progress bar
loss_history = []

# data batching 
num_sequences = int(train_input.shape[0] / seq_length)

# Run training
for epoch in range(n_epochs):    
    model.train()
    device = next(model.parameters()).device
    print("\n+ Epoch : ", epoch)
    total_loss = 0
    for seq_id in range(num_sequences):
        start_idx = seq_id*seq_step
        end_idx   = seq_id*seq_step + seq_length 
        # print(seq_id, start_idx, end_idx)
        
        input_seq_batch  = train_input[start_idx:end_idx].unsqueeze(0).to(device)
        target_seq_batch = train_target[start_idx:end_idx].unsqueeze(0).to(device)        
        optimizer.zero_grad()

        # Compute prediction and loss
        sos = model(input_seq_batch)
        y = lfilter(waveform=input_seq_batch, b_coeffs=sos[:, :3], a_coeffs=sos[:, 3:])
        batch_loss = torch.nn.functional.mse_loss(y, target_seq_batch)
        
        sos.requires_grad_(True)
        y.requires_grad_(True)
        batch_loss.requires_grad_(True)

        print("|-> y                            : ", y.grad)
        print("|-> sos                          : ", sos.grad)
        print("|-> batch_loss (before backprop) : ", batch_loss.grad)

        # Backpropagation

        batch_loss.backward()
        print("|-> batch_loss (after backprop)  : ", batch_loss.grad)

        optimizer.step()
        total_loss += batch_loss.item()
        print(f"|=========> Sequence {seq_id}: Loss = {batch_loss.item():.9f}")
    
        
    # record loss
    epoch_loss = total_loss / num_sequences
    loss_history.append(epoch_loss)
    print("-"* 100)
    print(f"|=========> epoch_loss = {epoch_loss:.3f} | Loss = {epoch_loss:.3f}")

    # Update the progress bar
    #pbar.set_description(f"\nEpoch: {epoch}, Loss: {epoch_loss:.9f}\n")
    #pbar.update(1)
    scheduler.step(total_loss)
    print("*"* 100)
    
# End timing the loop & print duration
elapsed_time = time.time() - start_time
print(f"\nOptimization loop took {elapsed_time:.2f} seconds.")

# Plot predicted filter
predicted_a = model.sos[:, 3:].detach().cpu().T.squeeze(1)
predicted_b = model.sos[:, :3].detach().cpu().T.squeeze(1)
freqs_hz, predicted_mag_response_db, predicted_phase_response_deg = model.compute_filter_magnitude_and_phase_frequency_response(model.get_dirac(fs, 0, grad=False), fs, predicted_a, predicted_b)