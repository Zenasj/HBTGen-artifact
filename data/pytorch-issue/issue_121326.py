import torch

#0 nf4_split /data/users/weif/transformer_nuggets/transformer_nuggets/quant/nf4_tensor.py:78
#1 __torch_dispatch__ /data/users/weif/transformer_nuggets/transformer_nuggets/quant/nf4_tensor.py:581
#2 PyObject_CallFunctionObjArgs /usr/local/src/conda/python-3.10.12/Objects/call.c:841
#3 torch::handle_torch_function_no_python_arg_parser(c10::ArrayRef<_object*>, _object*, _object*, char const*, _object*, char const*, torch::TorchFunctionName) ??:0
#4 (anonymous namespace)::ConcretePyInterpreterVTable::dispatch(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) const PyInterpreter.cpp:0
#5 (anonymous namespace)::pythonFallback(c10::OperatorHandle const&, std::vector<c10::IValue, std::allocator<c10::IValue> >*) PythonFallbackKernel.cpp:0
#6 c10::impl::BoxedKernelWrapper<std::vector<at::Tensor, std::allocator<at::Tensor> > (at::Tensor const&, c10::SymInt, long), void>::call(c10::BoxedKernel const&, c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) :0
#7 at::_ops::split_Tensor::redispatch(c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) ??:0
#8 torch::ADInplaceOrView::(anonymous namespace)::split_Tensor(c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) ADInplaceOrViewType_0.cpp:0
#9 c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<std::vector<at::Tensor, std::allocator<at::Tensor> > (c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long), &torch::ADInplaceOrView::(anonymous namespace)::split_Tensor>, std::vector<at::Tensor, std::allocator<at::Tensor> >, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long> >, std::vector<at::Tensor, std::allocator<at::Tensor> > (c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long)>::call(c10::OperatorKernel*, c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) ADInplaceOrViewType_0.cpp:0
#10 at::_ops::split_Tensor::redispatch(c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) ??:0
#11 torch::autograd::VariableType::(anonymous namespace)::split_Tensor(c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) VariableType_0.cpp:0
#12 c10::impl::make_boxed_from_unboxed_functor<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<std::vector<at::Tensor, std::allocator<at::Tensor> > (c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long), &torch::autograd::VariableType::(anonymous namespace)::split_Tensor>, std::vector<at::Tensor, std::allocator<at::Tensor> >, c10::guts::typelist::typelist<c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long> >, false>::call(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*) VariableType_0.cpp:0
#13 void c10::BoxedKernel::make_boxed_function<&(anonymous namespace)::pythonTLSSnapshotFallback>(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*) PythonFallbackKernel.cpp:0
#14 c10::impl::BoxedKernelWrapper<std::vector<at::Tensor, std::allocator<at::Tensor> > (at::Tensor const&, c10::SymInt, long), void>::call(c10::BoxedKernel const&, c10::OperatorHandle const&, c10::DispatchKeySet, at::Tensor const&, c10::SymInt, long) :0
#15 at::_ops::split_Tensor::call(at::Tensor const&, c10::SymInt, long) ??:0
#16 at::native::chunk(at::Tensor const&, long, long) ??:0
#17 c10::impl::make_boxed_from_unboxed_functor<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<std::vector<at::Tensor, std::allocator<at::Tensor> > (at::Tensor const&, long, long), &at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__chunk>, std::vector<at::Tensor, std::allocator<at::Tensor> >, c10::guts::typelist::typelist<at::Tensor const&, long, long> >, false>::call(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*) RegisterCompositeImplicitAutograd.cpp:0
#18 void c10::BoxedKernel::make_boxed_function<&(anonymous namespace)::pythonTLSSnapshotFallback>(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*) PythonFallbackKernel.cpp:0
#19 at::_ops::chunk::call(at::Tensor const&, long, long) ??:0
#20 torch::autograd::THPVariable_chunk(_object*, _object*, _object*) python_torch_functions_2.cpp:0
#21 cfunction_call /usr/local/src/conda/python-3.10.12/Objects/methodobject.c:543
#22 _chunk_with_empty /home/weif/local/pytorch-official/pytorch/torch/distributed/_composable/fsdp/_fsdp_common.py:97
#23 _init_sharded_param /home/weif/local/pytorch-official/pytorch/torch/distributed/_composable/fsdp/_fsdp_param.py:217
#24 decorate_context /home/weif/local/pytorch-official/pytorch/torch/utils/_contextlib.py:115
#25 __init__ /home/weif/local/pytorch-official/pytorch/torch/distributed/_composable/fsdp/_fsdp_param.py:141
#26 type_call /usr/local/src/conda/python-3.10.12/Objects/typeobject.c:1135
#27 <listcomp> /home/weif/local/pytorch-official/pytorch/torch/distributed/_composable/fsdp/_fsdp_param_group.py:94

from torch.utils._traceback import CapturedTraceback
print(''.join(CapturedTraceback.extract(cpp=True).format()))