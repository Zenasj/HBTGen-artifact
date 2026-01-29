# torch.rand(4, 128, 512, dtype=torch.float32)
import torch
import torch.nn as nn

# Define custom forward operator
torch.library.define("knl::fwd", "(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor, Tensor)")

@torch.library.impl("knl::fwd", "default")
def cuda_knl_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return (q.clone(), k.clone(), v.clone())

@torch.library.impl_abstract("knl::fwd", cuda_knl_fwd)
def meta_knl_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))

# Define custom backward operator
torch.library.define("knl::bwd", "(Tensor dout, Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor, Tensor)")

@torch.library.impl("knl::bwd", "default")
def cuda_knl_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return (q.clone(), k.clone(), v.clone())

@torch.library.impl_abstract("knl::bwd", cuda_knl_bwd)
def meta_knl_bwd(dout: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return (torch.empty_like(q), torch.empty_like(k), torch.empty_like(v))

class CallKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        q_padded, k_padded, v_padded = torch.ops.knl.fwd(q, k, v)
        ctx.save_for_backward(q_padded, k_padded, v_padded)
        return q_padded  # Forward output is first tensor from custom op

    @staticmethod
    def backward(ctx, dout):
        q_padded, k_padded, v_padded = ctx.saved_tensors
        dq, dk, dv = torch.ops.knl.bwd(dout, q_padded, k_padded, v_padded)
        return dq, dk, dv

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 512
        self.n_heads = 8
        self.kv_dim = 64
        self.inner_dim = self.n_heads * self.kv_dim  # 512
        self.Wq = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.Wk = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.Wv = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    def forward(self, hidden_states):
        q = self.Wq(hidden_states)
        k = self.Wk(hidden_states)
        v = self.Wv(hidden_states)
        output = CallKernel.apply(q, k, v)
        return self.o(output)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(4, 128, 512, dtype=torch.float32, device='cuda')

