"""
from chatgpt
"""
import torch.cuda.nvtx
from contextlib import contextmanager

@contextmanager
def nvtx_context(message):
    try:
        torch.cuda.nvtx.range_push(message)
        yield
    finally:
        torch.cuda.nvtx.range_pop()

# Example usage:
with nvtx_context("my_message"):
    # Code block where NVTX range is active
    pass