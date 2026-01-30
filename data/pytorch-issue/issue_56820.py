import numpy as np

test_reference_nd_fft_ifftn_cuda_complex64 (__main__.TestFFTCUDA)
test_reference_nd_fft_ifftn_cuda_float32 (__main__.TestFFTCUDA)
test_reference_nd_fft_irfftn_cuda_complex64 (__main__.TestFFTCUDA)
test_reference_nd_fft_irfftn_cuda_float32 (__main__.TestFFTCUDA)

def test_reference_nd(self, device, dtype, op):
        norm_modes = ((None, "forward", "backward", "ortho")
                      if LooseVersion(np.__version__) >= '1.20.0'
                      else (None, "ortho"))