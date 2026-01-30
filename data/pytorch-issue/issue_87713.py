def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
    r"""Performs dense-dense matrix multiplication according to segments along
    the first dimension of :obj:`inputs` as given by :obj:`ptr`, utilizing
    dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = torch.randn(8, 16)
        ptr = torch.tensor([0, 5, 8])
        other = torch.randn(2, 16, 32)

        out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
        assert out.size() == (8, 32)
        assert out[0:5] == inputs[0:5] @ other[0]
        assert out[5:8] == inputs[5:8] @ other[1]

    Args:
        input (torch.Tensor): The left operand 2D matrix of shape
            :obj:`[N, K]`.
        ptr (torch.Tensor): Compressed vector of shape :obj:`[B + 1]`, holding
            the boundaries of segments.
            For best performance, given as a CPU tensor.
        other (torch.Tensor): The right operand 3D tensor of shape
            :obj:`[B, K, M]`.

    Returns:
        torch.Tensor: The 2D output matrix of shape :obj:`[N, M]`.
    """

def grouped_matmul(inputs: List[Tensor], others: List[Tensor]) -> List[Tensor]:
    r"""Performs dense-dense matrix multiplication according to groups,
    utilizing dedicated kernels that effectively parallelize over groups.

    .. code-block:: python

        inputs = [torch.randn(5, 16), torch.randn(3, 32)]
        others = [torch.randn(16, 32), torch.randn(32, 64)]

        outs = pyg_lib.ops.grouped_matmul(inputs, others)
        assert len(outs) == 2
        assert outs[0].size() == (5, 32)
        assert outs[0] == inputs[0] @ others[0]
        assert outs[1].size() == (3, 64)
        assert outs[1] == inputs[1] @ others[1]

    Args:
        inputs (List[torch.Tensor]): List of left operand 2D matrices of shapes
            :obj:`[N_i, K_i]`.
        others (List[torch.Tensor]): List of right operand 2D matrices of
            shapes :obj:`[K_i, M_i]`.

    Returns:
        List[torch.Tensor]: List of 2D output matrices of shapes
        :obj:`[N_i, M_i]`.
    """

import pytest
import torch

import pyg_lib


def assert_close_enough(x, y, tol=1e-5):
    assert ((x - y).abs().max() <= tol), 'Max Abs Err: ' + str(float(
        (x - y).abs().max())) + ', Tolerance: ' + str(tol)

def test_grouped_matmul_autograd():
    device_str = 'cuda:0'
    device = torch.device(device_str)
    inputs = [torch.randn(5, 16).to(device), torch.randn(6, 20).to(device), torch.randn(20, 18).to(device)]
    others = [
        torch.randn((16, 48), requires_grad=True, device=device_str),
        torch.randn((20, 48), requires_grad=True, device=device_str),
        torch.randn((18, 49), requires_grad=True, device=device_str)
    ]
    outs = pyg_lib.ops.grouped_matmul(inputs, others)
    for out in outs:
        assert out.requires_grad
    assert len(outs) == len(inputs)
    
    sum([out.sum() for out in outs]).backward()
    for i in range(len(outs)):
        assert others[i].grad.shape == others[i].shape

import pytest
import torch

import pyg_lib

DEVICE_STRS = ['cuda:0']
major_vers, minor_vers = str(torch.__version__).split('.')[:2]
test_group_matmul = int(major_vers) >= 2 or int(minor_vers) >= 14
if int(major_vers) >= 2 or int(minor_vers) >= 12: # This only exists after 1.12
    torch.set_float32_matmul_precision('highest') # Enforce FP32
torch.backends.cuda.matmul.allow_tf32 = False

def assert_close_enough(x, y, tol=1e-5):
    assert ((x - y).abs().max() <= tol), 'Max Abs Err: ' + str(float(
        (x - y).abs().max())) + ', Tolerance: ' + str(tol)


@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_segment_matmul_autograd(device_str):
    inputs = torch.randn((8, 16), requires_grad=True, device=device_str)
    ptr = torch.tensor([0, 5, 8]).to(torch.device(device_str))
    other = torch.randn((2, 16, 32), requires_grad=True, device=device_str)
    out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
    assert out.shape == (inputs.shape[0], other.shape[-1])
    assert_close_enough(out[0:ptr[1]], inputs[0:ptr[1]] @ other[0])
    assert_close_enough(out[ptr[1]:ptr[2]], inputs[ptr[1]:ptr[2]] @ other[1])
    out.sum().backward()
    assert other.grad.shape == other.shape
    assert inputs.grad.shape == inputs.shape

@pytest.mark.skipif(test_group_matmul, reason="grouped_matmul requires torch >= 1.14")
@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_grouped_matmul_autograd(device_str):
    device = torch.device(device_str)
    inputs = [torch.randn(5, 16).to(device), torch.randn(6, 9).to(device), torch.randn(3, 32).to(device)]
    others = [
        torch.randn((16, 48), requires_grad=True, device=device_str),
        torch.randn((9, 42), requires_grad=True, device=device_str),
        torch.randn((32, 64), requires_grad=True, device=device_str)
    ]
    outs = pyg_lib.ops.grouped_matmul(inputs, others)
    assert len(outs) == len(inputs)
    for i in range(len(outs)):
        assert outs[i].size() == (inputs[i].shape[0], others[i].shape[-1])
        assert_close_enough(outs[i], inputs[i] @ others[i])
    
    sum([out.sum() for out in outs]).backward()
    for i in range(len(outs)):
        assert others[i].grad.shape == others[i].shape

def grouped_matmul(inputs: List[Tensor], others: List[Tensor]) -> List[Tensor]:
    major_vers, minor_vers = str(torch.__version__).split('.')[:2]
    assert int(major_vers) >= 2 or int(minor_vers) >= 14, (
        'grouped_matmul only available w/ torch >= 1.14.0')
    inputs = torch.nested.as_nested_tensor(inputs).contiguous()
    others = torch.nested.as_nested_tensor(others).contiguous()
    return list(torch.bmm(inputs, others).contiguous().unbind())


def segment_matmul(inputs: Tensor, ptr: Tensor, other: Tensor) -> Tensor:
    major_vers, minor_vers = str(torch.__version__).split('.')[:2]
    if int(major_vers) >= 2 or int(minor_vers) >= 14:
        inputs = torch.nested.as_nested_tensor(
            list(inputs.split((ptr[1:] - ptr[:-1]).tolist()))).contiguous()
        others = torch.nested.as_nested_tensor([x for x in other]).contiguous()
        out = torch.cat(torch.bmm(inputs, others).contiguous().unbind())
        return out
    else:
        return torch.ops.pyg.segment_matmul(inputs, ptr, other)

import pytest
import torch

import pyg_lib

DEVICE_STRS = ['cuda:0']
major_vers, minor_vers = str(torch.__version__).split('.')[:2]
print(major_vers, minor_vers)
test_group_matmul = int(major_vers) >= 2 or int(minor_vers) >= 14
if int(major_vers) >= 2 or int(minor_vers) >= 12: # This only exists after 1.12
    print("gifhest")
    torch.set_float32_matmul_precision('highest') # Enforce FP32
torch.backends.cuda.matmul.allow_tf32 = False

def assert_close_enough(x, y, tol=1e-5):
    assert ((x - y).abs().max() <= tol), 'Max Abs Err: ' + str(float(
        (x - y).abs().max())) + ', Tolerance: ' + str(tol)


@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_segment_matmul_autograd(device_str):
    inputs = torch.randn((8, 16), requires_grad=True, device=device_str)
    ptr = torch.tensor([0, 5, 8]).to(torch.device(device_str))
    other = torch.randn((2, 16, 32), requires_grad=True, device=device_str)
    out = pyg_lib.ops.segment_matmul(inputs, ptr, other)
    assert out.shape == (inputs.shape[0], other.shape[-1])
    assert_close_enough(out[0:ptr[1]], inputs[0:ptr[1]] @ other[0])
    assert_close_enough(out[ptr[1]:ptr[2]], inputs[ptr[1]:ptr[2]] @ other[1])
    out.sum().backward()
    assert other.grad.shape == other.shape
    assert inputs.grad.shape == inputs.shape

@pytest.mark.skipif(not test_group_matmul, reason="grouped_matmul requires torch >= 1.14")
@pytest.mark.parametrize('device_str', DEVICE_STRS)
def test_grouped_matmul_autograd(device_str):
    device = torch.device(device_str)
    inputs = [torch.randn(5, 16).to(device), torch.randn(6, 9).to(device), torch.randn(3, 32).to(device)]
    others = [
        torch.randn((16, 48), requires_grad=True, device=device_str),
        torch.randn((9, 42), requires_grad=True, device=device_str),
        torch.randn((32, 64), requires_grad=True, device=device_str)
    ]
    outs = pyg_lib.ops.grouped_matmul(inputs, others)
    assert len(outs) == len(inputs)
    # for i in range(len(outs)):
    #     assert outs[i].size() == (inputs[i].shape[0], others[i].shape[-1])
    #     assert_close_enough(outs[i], inputs[i] @ others[i])
    
    sum([out.sum() for out in outs]).backward()
    for i in range(len(outs)):
        assert others[i].grad.shape == others[i].shape