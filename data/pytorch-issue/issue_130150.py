import torch
from typing import Optional, List
from torch.library import custom_op
from mamba_ssm.ops.selective_scan_interface import selective_scan_cuda


###########################################
# Custom Operator Registration
###########################################

@custom_op(
    "mamba_selective_scan::selective_scan_fwd",
    mutates_args=(),
    device_types="cuda",
)
def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    Forward operator registration using the bridge.
    """
    return selective_scan_fwd_bridge(u, delta, A, B, C, D, z, delta_bias, delta_softplus)


@custom_op(
    "mamba_selective_scan::selective_scan_bwd",
    mutates_args=(),
    device_types="cuda",
)
def selective_scan_bwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    dout: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Backward operator registration using the bridge.
    """
    return selective_scan_bwd_bridge(u, delta, A, B, C, D, z, delta_bias, dout)


###########################################
# FakeTensor Implementations
###########################################

@selective_scan_fwd.register_fake
def selective_scan_fwd_fake(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,
) -> torch.Tensor:
    """
    FakeTensor implementation for selective_scan_fwd.
    """
    if D is None:
        D = torch.zeros_like(A[:, 0], device=u.device)
    if z is None:
        z = torch.zeros_like(u, device=u.device)

    # Return a tensor with the correct shape, dtype, and device
    return torch.empty_like(u)


@selective_scan_bwd.register_fake
def selective_scan_bwd_fake(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    dout: torch.Tensor,
) -> List[torch.Tensor]:
    """
    FakeTensor implementation for selective_scan_bwd.
    """
    if D is None:
        D = torch.zeros_like(A[:, 0], device=u.device)
    if z is None:
        z = torch.zeros_like(u, device=u.device)

    # Return gradients as FakeTensors
    return [torch.empty_like(t) if t is not None else torch.zeros(0, device=u.device) for t in [u, delta, A, B, C, D, z, delta_bias]]


###########################################
# Forward Bridge
###########################################
def selective_scan_fwd_bridge(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = False,  # Boolean input
) -> torch.Tensor:
    """
    Forward bridge for selective scan.
    """
    # Replace None values with default tensors
    if D is None:
        D = torch.zeros(A.size(0), device=u.device, dtype=u.dtype)
    if z is None:
        z = torch.ones_like(u)
    if delta_bias is None:
        delta_bias = torch.zeros(A.size(0), device=u.device, dtype=u.dtype)

    # Ensure contiguity
    tensors = [u, delta, A, B, C, D, z, delta_bias]
    tensors = [t.contiguous() for t in tensors if t is not None]
    u, delta, A, B, C, D, z, delta_bias = tensors

    # Convert delta_softplus to a scalar (1.0 for True, 0.0 for False)
    delta_softplus_scalar = 1.0 if delta_softplus else 0.0

    # Call the CUDA kernel
    return selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus_scalar)

###########################################
# Backward Bridge
###########################################

def selective_scan_bwd_bridge(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: Optional[torch.Tensor],
    z: Optional[torch.Tensor],
    delta_bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    delta_softplus: bool = False,  # Boolean input
) -> List[torch.Tensor]:
    """
    Backward bridge for selective scan.
    """
    # Replace None values with default tensors
    if D is None:
        D = torch.zeros(A.size(0), device=u.device, dtype=u.dtype)
    if z is None:
        z = torch.ones_like(u)
    if delta_bias is None:
        delta_bias = torch.zeros(A.size(0), device=u.device, dtype=u.dtype)

    # Ensure contiguity
    tensors = [u, delta, A, B, C, D, z, delta_bias, dout]
    tensors = [t.contiguous() for t in tensors if t is not None]
    u, delta, A, B, C, D, z, delta_bias, dout = tensors

    # Convert delta_softplus to a scalar (1.0 for True, 0.0 for False)
    delta_softplus_scalar = 1.0 if delta_softplus else 0.0

    # Call the CUDA kernel
    return selective_scan_cuda.bwd(u, delta, A, B, C, D, z, delta_bias, dout, delta_softplus_scalar)

###########################################
# Autograd Function
###########################################

class SelectiveScanAutogradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        # Call the forward custom op
        out = selective_scan_fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)

        # Save Tensors for backward
        ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias)

        # Save non-Tensor arguments
        ctx.delta_softplus = delta_softplus
        return out

    @staticmethod
    def backward(ctx, dout):
        # Retrieve saved tensors
        u, delta, A, B, C, D, z, delta_bias = ctx.saved_tensors

        # Call the backward custom op
        gradients = selective_scan_bwd(u, delta, A, B, C, D, z, delta_bias, dout)

        # Unpack gradients for Tensor arguments
        du, ddelta, dA, dB, dC, dD, dz, ddelta_bias = gradients

        # Explicitly set gradients for optional arguments to None if they were None in the forward pass
        dD = dD if D is not None else None
        dz = dz if z is not None else None
        ddelta_bias = ddelta_bias if delta_bias is not None else None

        # Explicitly set gradient for non-Tensor arguments to None
        return du, ddelta, dA, dB, dC, dD, dz, ddelta_bias, None

###########################################
# User-Facing Function
###########################################

def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    """
    User-facing function for selective scan.
    """
    return SelectiveScanAutogradFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus)