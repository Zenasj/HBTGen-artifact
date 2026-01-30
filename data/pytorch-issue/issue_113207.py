frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x95 (0x7fa21b99d975 in /data/users/ezyang/c/pytorch/torch/lib/libc10.so)                                                                                                                                                                                                         
frame #1: c10::TensorImpl::throw_cannot_call_with_symbolic(char const*) const + 0x8d (0x7fa21b951269 in /data/users/ezyang/c/pytorch/torch/lib/libc10.so)                                                  
frame #2: c10::TensorImpl::sizes_custom() const + 0x9f (0x7fa21b9770df in /data/users/ezyang/c/pytorch/torch/lib/libc10.so)                                                                                
frame #3: at::meta::structured_mm::meta(at::Tensor const&, at::Tensor const&) + 0x31e (0x7fa20a202a8e in /data/users/ezyang/c/pytorch/torch/lib/libtorch_cpu.so)                                           
frame #4: <unknown function> + 0x29f34de (0x7fa20b5f34de in /data/users/ezyang/c/pytorch/torch/lib/libtorch_cpu.so)                                                                                        
frame #5: <unknown function> + 0x2a1fd8e (0x7fa20b61fd8e in /data/users/ezyang/c/pytorch/torch/lib/libtorch_cpu.so)                                                                                        
frame #6: <unknown function> + 0x6b907b (0x7fa2142b907b in /data/users/ezyang/c/pytorch/torch/lib/libtorch_python.so)                                                                                      
frame #7: <unknown function> + 0x6b6175 (0x7fa2142b6175 in /data/users/ezyang/c/pytorch/torch/lib/libtorch_python.so)

#4 c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) from ??:0                                                                       
#5 c10::TensorImpl::throw_cannot_call_with_symbolic(char const*) const from ??:0
#6 c10::TensorImpl::sizes_custom() const [clone .localalias] from TensorImpl.cpp:0
#7 at::meta::structured_mm::meta(at::Tensor const&, at::Tensor const&) from ??:0
#8 at::(anonymous namespace)::wrapper_Meta_mm_out_out(at::Tensor const&, at::Tensor const&, at::Tensor&) from RegisterMeta.cpp:0
#9 c10::impl::make_boxed_from_unboxed_functor<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor& (at::Tensor const&, at::Tensor const&, at::Tensor&), &at::(anonymous namespace)::wrapper_Meta_mm_out_out>, at::Tensor&, c10::guts::typelist::typelist<at::Tensor const&, at::Tensor const&, at::Tensor&> >, false>::call(c10::OperatorKernel*, c10::OperatorHandle const&, c10::DispatchKeySet, std::vector<c10::IValue, std::allocator<c10::IValue> >*) from RegisterMeta.cpp:0

import torch
import torch.distributed as dist
if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    dist.all_gather_object([None, None], dist.get_rank())

import torch
import torch.distributed as dist
from torch.distributed._tensor import distribute_tensor, Shard
from torch.distributed.device_mesh import init_device_mesh
if __name__ == '__main__':
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    t = torch.randn(256, device='cuda')
    mesh = init_device_mesh('cuda', (2, ))
    dt = distribute_tensor(t, mesh, [Shard(0)])