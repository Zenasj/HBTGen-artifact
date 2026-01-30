import torch

py
import sys
from unittest.mock import Mock, patch

docstrings = {}


def gather_docstrings(func, docstr):
    docstrings[func._extract_mock_name()] = docstr


with patch.dict(sys.modules, {"torch": Mock(name="torch"), "torch._C": Mock(_add_docstr=gather_docstrings)}):
    sys.path.append("torch")  # bypassing torch/__init__.py
    import _torch_docs

print(docstrings)
# result: {'torch.abs': '\nabs(input, out=None) -> Tensor...', ...}