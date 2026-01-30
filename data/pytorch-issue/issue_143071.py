import torch.nn as nn
import numpy as np

import torch
import math

x = torch.tensor(2.0, requires_grad=True)
y = math.exp(x)
"""
RuntimeError: math.exp is not currently a tensor supported operation. Please use torch.exp instead.
"""

import torch
from copy import deepcopy

x = torch.tensor(2.0, requires_grad=True)
x = x+1
y = deepcopy(x)
"""
RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001
"""

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2

x = torch.tensor(2.0, requires_grad=True)
y = math.pow(x,2)

x = torch.tensor(2.0, requires_grad=True)
y = np.pow(x,2)
# > RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

import torch

x = torch.tensor(2.0, requires_grad=True)
x

"""
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:145: UserWarning: 1a Tensor with 1 elements cannot be converted to Scalar (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  nonzero_finite_vals = torch.masked_select(
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:159: UserWarning: 1a Tensor with 1 elements cannot be converted to Scalar (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  if value != torch.ceil(value):
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:167: UserWarning: 1a Tensor with 1 elements cannot be converted to Scalar (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  nonzero_finite_max / nonzero_finite_min > 1000.0
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:168: UserWarning: 1a Tensor with 1 elements cannot be converted to Scalar (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  or nonzero_finite_max > 1.0e8
/Users/joshuahamilton/open-source/pytorch/torch/_tensor.py:1097: UserWarning: 1a Tensor with 1 elements cannot be converted to Scalar (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  return self.item().__format__(format_spec)
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:225: UserWarning: 1a Tensor with 1 elements cannot be converted to Scalar (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  return formatter1.format(self.item())

tensor(2., requires_grad=True)
"""

import torch

x = torch.tensor(2.0, requires_grad=True)
x
"""
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:145: UserWarning: 1warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  nonzero_finite_vals = torch.masked_select(
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:159: UserWarning: 1warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  if value != torch.ceil(value):
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:167: UserWarning: 1warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  nonzero_finite_max / nonzero_finite_min > 1000.0
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:168: UserWarning: 1warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  or nonzero_finite_max > 1.0e8
/Users/joshuahamilton/open-source/pytorch/torch/_tensor.py:1097: UserWarning: 1warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  return self.item().__format__(format_spec)
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:225: UserWarning: 1warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)

tensor(2., requires_grad=True)
"""

import torch

x = torch.tensor(2.0, requires_grad=True)
x
"""
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:145: UserWarning: 1still a warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  nonzero_finite_vals = torch.masked_select(
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:159: UserWarning: 1still a warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  if value != torch.ceil(value):
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:167: UserWarning: 1still a warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  nonzero_finite_max / nonzero_finite_min > 1000.0
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:168: UserWarning: 1still a warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  or nonzero_finite_max > 1.0e8
/Users/joshuahamilton/open-source/pytorch/torch/_tensor.py:1097: UserWarning: 1still a warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  return self.item().__format__(format_spec)
/Users/joshuahamilton/open-source/pytorch/torch/_tensor_str.py:225: UserWarning: 1still a warning message! (Triggered internally at /Users/joshuahamilton/open-source/pytorch/aten/src/ATen/native/Scalar.cpp:18.)
  return formatter1.format(self.item())

tensor(2., requires_grad=True)
"""