import torch
import torch.nn as nn

lin = nn.Linear(10, 10)
x = torch.randn(1, 10)

with torch.autocast(enabled=False, dtype=torch.float16, device_type="cpu"):
    out = lin(x)
# RuntimeError: Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype

class AutocastWrapper:
    """
    There is a PyTorch issue requiring bfloat16 support, even if we have AMP disabled
    Workaround with a wrapper: https://github.com/pytorch/pytorch/pull/9609
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __enter__(self):
        if self.enabled and torch.cuda.is_bf16_supported():
            self.autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)
            self.autocast.__enter__()
        elif self.enabled:
            logging.warning("device does not support bfloat16, disabling AMP")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.autocast.__exit__(exc_type, exc_val, exc_tb)