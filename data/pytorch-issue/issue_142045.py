import torch
from enum import Enum

# torch.rand(8192, 8192, dtype=torch.bfloat16), torch.rand(8192, 8192, dtype=torch.bfloat16)
class MyModel(torch.nn.Module):
    def __init__(self, scaling_strategy, fp8_kernel):
        super().__init__()
        self.scaling_strategy = scaling_strategy
        self.fp8_kernel = fp8_kernel  # Unused but preserved for configuration

    def forward(self, inputs):
        A, B = inputs
        A_fp8 = A.to(torch.float8_e4m3fn)
        B_fp8 = B.to(torch.float8_e4m3fn)
        # Preprocess stub (from torchao.float8.inference)
        def preprocess_data_stub(a, b, config):
            return a, b  # Placeholder for actual preprocessing logic
        A_fp8, B_fp8 = preprocess_data_stub(A_fp8, B_fp8, None)  # Assume Float8MMConfig
        
        if self.scaling_strategy == ScalingStrategy.PER_TENSOR:
            a_scale = torch.tensor(1.0, device=A.device, dtype=torch.float32)
            b_scale = torch.tensor(1.0, device=B.device, dtype=torch.float32)
        elif self.scaling_strategy == ScalingStrategy.PER_ROW:
            a_scale = torch.ones((A.size(0), 1), device=A.device, dtype=torch.float32)
            b_scale = torch.ones((B.size(1), 1), device=B.device, dtype=torch.float32).transpose(0, 1)
        else:
            raise ValueError("Invalid scaling strategy")
        
        # Stub for addmm_float8_unwrapped_inference (from torchao.float8.inference)
        def addmm_float8_stub(a, a_scale, b, b_scale, output_dtype, **kwargs):
            return torch.matmul(a.to(torch.float32), b.to(torch.float32)).to(output_dtype)
        return addmm_float8_stub(
            A_fp8, a_scale, B_fp8, b_scale, output_dtype=torch.bfloat16, use_fast_accum=True
        )

# Enums as defined in the PR's code
class ScalingStrategy(Enum):
    PER_TENSOR = "PerTensor"
    PER_ROW = "PerRow"

class FP8Kernel(Enum):
    SCALED_MM = "Scaled-MM"

def my_model_function():
    return MyModel(scaling_strategy=ScalingStrategy.PER_TENSOR, fp8_kernel=FP8Kernel.SCALED_MM)

def GetInput():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    B = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    return (A, B)

