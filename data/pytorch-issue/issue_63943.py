import torch

ref_array = torch.randn(int(1e4) + 1)
test_array1 = torch.Tensor(ref_array)
test_array2 = torch.Tensor(ref_array)
test_array3 = torch.Tensor(ref_array)

# recovering input array works for fft -> ifft for array of uneven len
test_array1 = torch.fft.fft(test_array1)
test_array1 = torch.fft.ifft(test_array1)
assert torch.allclose(ref_array, torch.real(test_array1), atol=1e-6)
# recovering input array works for rfft -> irfft for array of even len
test_array2 = torch.fft.rfft(test_array2[:int(1e4)])
test_array2 = torch.fft.irfft(test_array2[:int(1e4)])
assert torch.allclose(ref_array[:int(1e4)], test_array2, atol=1e-6)
# recovering input array fails for rfft -> irfft for array of odd len
test_array3 = torch.fft.rfft(test_array3)
test_array3 = torch.fft.irfft(test_array3)
# we get size error and the array does not match
try:
    assert torch.allclose(ref_array, test_array3, atol=1e-6)
except RuntimeError:
    assert not torch.allclose(ref_array[:int(1e4)], test_array3[:int(1e4)], atol=1e-1)