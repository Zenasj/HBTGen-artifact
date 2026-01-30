import pytest
import torch

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("m", [32, 64])
@pytest.mark.parametrize("k", [32, 64])
@pytest.mark.parametrize("n", [32, 64])
def test__int_mm(device, m, k, n):

    def genf_int_float(x, y):
        x_int8 = torch.randint(-128, 128, (x, y), dtype=torch.int8, device=device)
        x_float = x_int8.to(torch.float32)
        return x_int8, x_float

    a_int8, a_float = genf_int_float(m, k)
    b_int8, b_float = genf_int_float(k, n)
    c_int32 = torch._int_mm(a_int8, b_int8)
    assert torch.equal(c_int32.float(), torch.mm(a_float, b_float))
    c_int32_result = c_int32.new_empty(c_int32.size())
    torch._int_mm(a_int8, b_int8, out=c_int32_result)
    assert torch.equal(c_int32_result.float(), torch.mm(a_float, b_float))