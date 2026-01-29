# torch.rand(3, 3, dtype=torch.float32)
import torch
import numpy as np
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute PyTorch SVD
        U_torch, S_torch, Vh_torch = torch.svd(x, some=False)
        # Compute NumPy SVD
        x_np = x.detach().cpu().numpy()
        U_np, S_np, Vh_np = np.linalg.svd(x_np, full_matrices=True)
        # Convert back to tensors on the same device
        device = x.device
        U_np = torch.from_numpy(U_np).to(device)
        S_np = torch.from_numpy(S_np).to(device)
        Vh_np = torch.from_numpy(Vh_np).to(device)
        # Compare S (should be identical)
        S_close = torch.allclose(S_torch, S_np, atol=1e-6, rtol=1e-6)
        # Compare U and V, allowing for sign flips and permutations
        # U comparison (minimize sign differences)
        U_diff = torch.min(
            torch.abs(U_torch - U_np),
            torch.abs(U_torch + U_np)
        ).max()
        # V comparison (Vh_torch is V^H; compare with Vh_np's transpose)
        V_torch = Vh_torch.t()
        V_np = Vh_np.t()
        V_diff = torch.min(
            torch.abs(V_torch - V_np),
            torch.abs(V_torch + V_np)
        ).max()
        # Return True if all within tolerance
        threshold = 1e-6
        return torch.tensor([S_close and (U_diff < threshold) and (V_diff < threshold)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Example matrix from the issue that triggers discrepancies
    R = torch.tensor([[0.41727819, -0.87345426,  0.25091147],
                      [0.32246181,  0.40043949,  0.85771009],
                      [-0.84964539, -0.27699435,  0.44875031]], dtype=torch.float32)
    return R

