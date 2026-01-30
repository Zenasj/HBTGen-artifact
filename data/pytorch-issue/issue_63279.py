import torch
t1 = torch.empty(())
t2 = torch.zeros((3,))

from . import _nn as _nn
from . import _onnx as _onnx
from . import _VariableFunctions as _VariableFunctions

from ._nn import *
from ._onnx import *
from ._VariableFunctions import *