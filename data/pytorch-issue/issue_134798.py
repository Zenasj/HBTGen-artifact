def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        from vllm import _custom_ops as ops

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(
            out,
            x,
            self.weight.data,   ##THIS ONE HERE 
            self.variance_epsilon,
        )
        return out

py
import torch

@torch._dynamo.allow_in_graph # E: expected 2 blank lines, found 1
def get_data(x):
    y = torch._C._autograd._get_data_attr(x) # E: Module has no attribute "_get_data_attr"  [attr-defined]
    return y

@torch.compile(fullgraph=True) # E: expected 2 blank lines, found 1
def f(x):
    # x = torch.randn(3)
    y = get_data(x)
    return y

with torch.inference_mode(): # E: expected 2 blank lines after class or function definition, found 1
    x = torch.randn(3)
    f(x)