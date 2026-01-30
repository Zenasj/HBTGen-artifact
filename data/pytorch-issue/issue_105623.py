import torch

torch.testing.assert_close(
                            actual,
                            expected,
                            rtol=rtol,
                            atol=atol,
                            equal_nan=True,
                            check_device=False,
                        )