import torch

def test_to(dtype):
    element_size = torch.ones(1, dtype=dtype).element_size()
    width = 32_768
    height = 2**32 // width // element_size

    lt_2to32 = torch.ones(width-1, height, dtype=dtype)
    eq_2to32 = torch.ones(width,   height, dtype=dtype)
    assert element_size * lt_2to32.numel()  < 2**32     # 2^31.9999 bytes
    assert element_size * eq_2to32.numel() == 2**32     # 2^32      bytes

    assert torch.all(lt_2to32.to("mps").to("cpu") == 1) # 2^31.9999 bytes -> all ones
    assert torch.all(eq_2to32.to("mps").to("cpu") == 0) # 2^32      bytes -> all zeros

    del lt_2to32, eq_2to32

test_to(torch.float16)
test_to(torch.float32)
test_to(torch.int32)
test_to(torch.int64)