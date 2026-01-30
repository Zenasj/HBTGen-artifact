import numpy as np

y_np = numpy.zeros((n_batch, ceildiv(n_time - frame_length + 1, frame_step), fft_length // 2 + 1), dtype="complex64")
window_np = numpy.hanning(frame_length + 1 - frame_length % 2)[:frame_length]
for b in range(y_np.shape[0]):
    for t in range(y_np.shape[1]):
        for f in range(y_np.shape[2]):
            for k in range(frame_length):
                y_np[b, t, f] += (
                        window_np[k]
                        * x[b, t * frame_step + k]
                        * numpy.exp(-2j * numpy.pi * f * k / fft_length))

def ceildiv(a, b):
    return -(-a // b)

y_np = numpy.zeros((n_batch, ceildiv(n_time - fft_length + 1, frame_step), fft_length // 2 + 1), dtype="complex64")
window_np = numpy.hanning(frame_length + 1)[:frame_length]
for b in range(y_np.shape[0]):
    for t in range(y_np.shape[1]):
        for f in range(y_np.shape[2]):
            for k in range(frame_length):
                y_np[b, t, f] += (
                        window_np[k]
                        * x[b, t * frame_step + k + (fft_length - frame_length) // 2]
                        * numpy.exp(-2j * numpy.pi * f * (k + (fft_length - frame_length) // 2) / fft_length))