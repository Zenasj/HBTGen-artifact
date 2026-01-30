import torch
arg_1 = torch.rand([5, 5], dtype=torch.float64)
arg_2 = torch.rand([5, 5], dtype=torch.float64)
arg_3 = torch.rand([1, 5], dtype=torch.complex128)
res = torch.addcmul(arg_1,arg_2,arg_3)
# RuntimeError: !(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ && (has_undefined_outputs || config.enforce_safe_casting_to_output_ || config.cast_common_dtype_to_outputs_))INTERNAL ASSERT FAILED at "../aten/src/ATen/TensorIterator.cpp":331, please report a bug to PyTorch.

import torch
arg_1 = torch.rand([1], dtype=torch.complex128)
arg_2 = torch.rand([5, 5, 1], dtype=torch.complex128)
arg_3 = torch.rand([1, 3], dtype=torch.float32)
res = torch.addcdiv(arg_1,arg_2,arg_3,)
# RuntimeError: !(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ && (has_undefined_outputs || config.enforce_safe_casting_to_output_ || config.cast_common_dtype_to_outputs_))INTERNAL ASSERT FAILED at "../aten/src/ATen/TensorIterator.cpp":331, please report a bug to PyTorch.

torch.ge
torch.le
torch.gt
torch.lt
torch.ne
torch.eq
torch.igamma
torch.igammac
torch.xlogy
torch.logical_and
torch.logical_or
torch.logical_xor
torch.bucketize
torch.searchsorted
torch.dist
torch.ldexp
torch.sub
torch.add
torch.mul
torch.div
torch.ger
torch.outer
torch.nextafter
torch.copysign
torch.pow

import torch
results = dict()
input_tensor = torch.rand([1], dtype=torch.float64)
input = input_tensor.clone()
other_tensor = torch.rand([1], dtype=torch.complex32)
other = other_tensor.clone()
try:
  results["res_1"] = torch.add(input, other, )
except Exception as e:
  results["err_1"] = "ERROR:"+str(e)
try:
  results["res_2"] = torch.sub(input_tensor.clone(),other_tensor.clone(),)
except Exception as e:
  results["err_2"] = "ERROR:"+str(e)

print(results)
# {'err_1': 'ERROR:common_dtype_ != ScalarType::UndefinedINTERNAL ASSERT FAILED at "../aten/src/ATen/TensorIterator.cpp":236, please report a bug to PyTorch. ', 'err_2': 'ERROR:common_dtype_ != ScalarType::UndefinedINTERNAL ASSERT FAILED at "../aten/src/ATen/TensorIterator.cpp":236, please report a bug to PyTorch. '}

import torch

# func_cls=torch.Tensor.addcmul
func_cls=torch.Tensor.addcdiv

arg_1 = torch.rand([5, 5], dtype=torch.float64)
arg_2 = torch.rand([5, 5], dtype=torch.float64)
arg_3 = torch.rand([1, 5], dtype=torch.complex128)

def test():
	tmp_result= func_cls(arg_1,arg_2,arg_3)
	return tmp_result
res = test()