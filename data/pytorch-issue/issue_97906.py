import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    # Display the current GPU name
    print("GPU name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available")

# Verify the PyTorch version
print("PyTorch version: ", torch.__version__)

import torch
print(torch.cuda.get_device_properties(0).multi_processor_count)