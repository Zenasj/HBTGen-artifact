import os
import torch
import torchaudio

# Set the environment variables to enable more information in error logs
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

# Define some tensors
a = torch.Tensor((1, 0))
b = torch.Tensor((1, 0))
waveform = torch.ones((1, 1, 16000))


# Define the function to compile
def my_lfilter(x):
    return torchaudio.functional.lfilter(x, a, b)


# Classic non compiled filtering
filtered_waveform = my_lfilter(waveform)

# Compiling the function, this is passing
my_lfilter_compiled = torch.compile(my_lfilter)

# The error is here, when calling the compiled function
filtered_waveform_compiled = my_lfilter_compiled(waveform)