import torch

# Set a deterministic seed
torch.manual_seed(42)

# Ensure MPS backend is available
if not torch.backends.mps.is_available():
    raise ValueError("MPS backend is not available on this machine.")

# Create a random tensor
A = torch.randn(9, 360000)

# Define the pairwise L1 distance function
def compute_l1_distances(tensor):
    N = tensor.shape[0]
    l1_distances = torch.empty(N, N)
    for i in range(N):
        l1_distances[i] = torch.sum(torch.abs(tensor[i] - tensor), dim=1)
    return l1_distances

# Compute on CPU
A_cpu = A.cpu()
l1_cpu = compute_l1_distances(A_cpu)
print("L1 distances (CPU):")
print(l1_cpu)

# Compute on MPS
A_mps = A.to(torch.device("mps"))
l1_mps = compute_l1_distances(A_mps)
print("\nL1 distances (MPS):")
print(l1_mps)

# Check for differences
diff = (l1_cpu.cpu() - l1_mps.cpu()).abs().max().item()
print(f"\nMaximum absolute difference: {diff}")