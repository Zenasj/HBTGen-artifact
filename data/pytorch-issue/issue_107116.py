# torch.rand(2, 3, 4, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        dims_to_test = [-3, -2, -1, 0, 1, 2]
        all_ok = True
        for dim in dims_to_test:
            try:
                mps_out = torch.std(x.to("mps"), dim=dim).cpu()
                cpu_out = torch.std(x, dim=dim)
                # Check both shape and numerical tolerance
                if (mps_out.shape != cpu_out.shape) or not torch.allclose(mps_out, cpu_out, atol=1e-4):
                    all_ok = False
                    break
            except Exception as e:
                # Catch any unexpected errors (e.g., invalid dim handling)
                all_ok = False
                break
        return torch.tensor([all_ok], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3, 4)

