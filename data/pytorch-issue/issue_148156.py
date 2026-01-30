import torch

# Create a random matrix on MPS
R = torch.tensor([[ 0.6047+1.1093j]], device='mps')
R_cpu = R.to("cpu")  # Copy to CPU

# Compute R^H R on both devices
mps_result = R.T.conj() @ R
cpu_result = R_cpu.T.conj() @ R_cpu

print("MPS Result:", mps_result)
print("CPU Result:", cpu_result)

torch.testing.assert_close(mps_result.to('cpu'), cpu_result)