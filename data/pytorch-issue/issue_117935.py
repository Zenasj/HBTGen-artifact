# test.py
import torch
import tempfile

def test_single():
    a = torch.ones((2, 2), device="cuda")