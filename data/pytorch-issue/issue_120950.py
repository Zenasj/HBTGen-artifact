import torch
import torch.nn as nn
import copy

torch.library.define("knl::fwd", "(Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor, Tensor)")

@torch.library.impl("knl::fwd", "default")
def cuda_knl_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    print(q.sum())
    print(k.sum())
    print(v.sum())

    return (q, k, v)

@torch.library.impl_abstract("knl::fwd", cuda_knl_fwd)
def meta_knl_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):

    return (torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v))

torch.library.define("knl::bwd", "(Tensor dout, Tensor q, Tensor k, Tensor v) -> (Tensor, Tensor, Tensor)")

@torch.library.impl("knl::bwd", "default")
def cuda_knl_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):
    print(q.sum())
    print(k.sum())
    print(v.sum())

    return (q, k, v)

@torch.library.impl_abstract("knl::bwd", cuda_knl_bwd)
def meta_knl_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
):

    return (torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v))

class CallKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):

        q_padded, k_padded, v_padded = torch.ops.knl.fwd(q, k, v)
        ctx.save_for_backward(q_padded, k_padded, v_padded)

        return q_padded

    @staticmethod
    def backward(ctx, dout, *args):
        q_padded, k_padded, v_padded = ctx.saved_tensors

        dq, dk, dv = torch.ops.knl.bwd(dout, q_padded, k_padded, v_padded)

        return dq, dk, dv


def call_kernel(q, k, v):
    return CallKernel.apply(q, k, v)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.d_model=512
        self.n_heads=8
        self.kv_dim=64
        self.inner_dim=self.n_heads * self.kv_dim

        self.Wq = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.Wk = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.Wv = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

    def forward(self, hidden_states):

        q = self.Wq(hidden_states)
        k = self.Wk(hidden_states)
        v = self.Wv(hidden_states)

        output = call_kernel(q, k, v)

        output = self.o(output)

        return output


model = Attention().cuda()
#model = nn.Sequential(Attention(), Attention()).cuda()
model_compiled = copy.deepcopy(model)
model_compiled = torch.compile(model_compiled)

input = torch.randn((4, 128, 512)).cuda()

print("Model fwd")
out_ref = model(input)
print("Model compiled fwd")
out = model_compiled(input)

print("Model bwd")
out_ref.sum().backward()
print("Model compiled bwd")
out.sum().backward()