import torch

test_non_contiguous_input_mm
test_non_contiguous_input_bmm
test_non_contiguous_input_addmm

self.assertTrue(torch.allclose(ref, act, atol=1e-2, rtol=1e-2))
AssertionError: False is not true