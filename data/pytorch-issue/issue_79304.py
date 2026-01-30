Hello0
Hello1
Hello3

Hello0
Hello1

#0  0x00007f5ee0233e7c in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#1  0x00007f5ee038d47b in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#2  0x00007f5ee0443d76 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#3  0x00007f5ee0269514 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#4  0x00007f5ee026967f in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#5  0x00007f5ee024335b in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#6  0x00007f5ee0243eea in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#7  0x00007f5ee0411583 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#8  0x00007f5ee0411627 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#9  0x00007f5ee0107d34 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#10 0x00007f5ee01107e0 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#11 0x00007f5ee0114a32 in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#12 0x00007f5ee0115e9c in ?? () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#13 0x00007f5ee01091ec in __cuda_CallJitEntryPoint () from /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1
#14 0x00007f5ee8621232 in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#15 0x00007f5ee865273d in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#16 0x00007f5ee83e5f2a in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#17 0x00007f5ee846061a in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#18 0x00007f5ee8460d9b in ?? () from /usr/lib/x86_64-linux-gnu/libcuda.so.1
#19 0x0000564a915ca984 in __cudart560 ()
#20 0x0000564a915bf8ae in __cudart604 ()
#21 0x0000564a915d10a4 in __cudart534 ()
#22 0x0000564a915d586a in __cudart778 ()
#23 0x0000564a915d5b04 in __cudart768 ()
#24 0x0000564a915c7a4c in __cudart941 ()
#25 0x0000564a915bb88c in __cudart1044 ()
#26 0x0000564a915fdd0b in cudaStreamIsCapturing ()
#27 0x0000564a81bbef29 in c10::cuda::currentStreamCaptureStatusMayInitCtx() ()
#28 0x0000564a81bac1b3 in c10::cuda::CUDACachingAllocator::(anonymous namespace)::cudaMallocMaybeCapturing(void**, unsigned long) ()
#29 0x0000564a81bae730 in c10::cuda::CUDACachingAllocator::DeviceCachingAllocator::alloc_block(c10::cuda::CUDACachingAllocator::(anonymous namespace)::AllocParams&, bool) ()
#30 0x0000564a81bac6f6 in c10::cuda::CUDACachingAllocator::DeviceCachingAllocator::malloc(int, unsigned long, CUstream_st*) ()
#31 0x0000564a81bc26f0 in c10::cuda::CUDACachingAllocator::THCCachingAllocator::malloc(void**, int, unsigned long, CUstream_st*) ()
#32 0x0000564a81bc349c in c10::cuda::CUDACachingAllocator::CudaCachingAllocator::allocate(unsigned long) const ()
#33 0x0000564a8606553f in at::detail::empty_generic(c10::ArrayRef<long>, c10::Allocator*, c10::DispatchKeySet, c10::ScalarType, c10::optional<c10::MemoryFormat>) ()
#34 0x0000564a85a55ed8 in at::detail::empty_cuda(c10::ArrayRef<long>, c10::ScalarType, c10::optional<c10::Device>, c10::optional<c10::MemoryFormat>) ()
#35 0x0000564a85a56013 in at::detail::empty_cuda(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#36 0x0000564a84a59fad in at::native::empty_cuda(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#37 0x0000564a85ba3bf1 in at::(anonymous namespace)::(anonymous namespace)::wrapper_memory_format_empty_memory_format(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#38 0x0000564a85d7d37d in c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>), &at::(anonymous namespace)::(anonymous namespace)::wrapper_memory_format_empty_memory_format>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat> > >, at::Tensor (c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#39 0x0000564a87cbe15d in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat> >(void*, c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>&&, c10::optional<c10::ScalarType>&&, c10::optional<c10::Layout>&&, c10::optional<c10::Device>&&, c10::optional<bool>&&, c10::optional<c10::MemoryFormat>&&) ()
#40 0x0000564a87c1695d in at::Tensor c10::Dispatcher::redispatch<at::Tensor, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)> const&, c10::DispatchKeySet, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) const ()
#41 0x0000564a87af90c5 in at::_ops::empty_memory_format::redispatch(c10::DispatchKeySet, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#42 0x0000564a86977bdd in at::(anonymous namespace)::empty_memory_format(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#43 0x0000564a8698bb4d in c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>), &at::(anonymous namespace)::empty_memory_format>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat> > >, at::Tensor (c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#44 0x0000564a87cbe15d in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat> >(void*, c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>&&, c10::optional<c10::ScalarType>&&, c10::optional<c10::Layout>&&, c10::optional<c10::Device>&&, c10::optional<bool>&&, c10::optional<c10::MemoryFormat>&&) ()
#45 0x0000564a87af8d22 in at::_ops::empty_memory_format::call(c10::ArrayRef<long>, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>, c10::optional<c10::MemoryFormat>) ()
#46 0x0000564a7e18887d in at::empty(c10::ArrayRef<long>, c10::TensorOptions, c10::optional<c10::MemoryFormat>) ()
#47 0x0000564a8660314a in at::native::full(c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#48 0x0000564a87030faf in at::(anonymous namespace)::(anonymous namespace)::wrapper__full(c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#49 0x0000564a871110f0 in c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>), &at::(anonymous namespace)::(anonymous namespace)::wrapper__full>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> > >, at::Tensor (c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#50 0x0000564a8800b0d1 in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> >(void*, c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>&&, c10::Scalar const&, c10::optional<c10::ScalarType>&&, c10::optional<c10::Layout>&&, c10::optional<c10::Device>&&, c10::optional<bool>&&) ()
#51 0x0000564a87fbcb3c in at::Tensor c10::Dispatcher::redispatch<at::Tensor, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> >(c10::TypedOperatorHandle<at::Tensor (c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)> const&, c10::DispatchKeySet, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) const ()
#52 0x0000564a87eed6a8 in at::_ops::full::redispatch(c10::DispatchKeySet, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#53 0x0000564a86978196 in at::(anonymous namespace)::full(c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#54 0x0000564a8698d6ea in c10::impl::wrap_kernel_functor_unboxed_<c10::impl::detail::WrapFunctionIntoFunctor_<c10::CompileTimeFunctionPointer<at::Tensor (c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>), &at::(anonymous namespace)::full>, at::Tensor, c10::guts::typelist::typelist<c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> > >, at::Tensor (c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>)>::call(c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#55 0x0000564a8800b0d1 in at::Tensor c10::callUnboxedKernelFunction<at::Tensor, c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool> >(void*, c10::OperatorKernel*, c10::DispatchKeySet, c10::ArrayRef<long>&&, c10::Scalar const&, c10::optional<c10::ScalarType>&&, c10::optional<c10::Layout>&&, c10::optional<c10::Device>&&, c10::optional<bool>&&) ()
#56 0x0000564a87eed30d in at::_ops::full::call(c10::ArrayRef<long>, c10::Scalar const&, c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>) ()
#57 0x0000564a7e15deee in at::full(c10::ArrayRef<long>, c10::Scalar const&, c10::TensorOptions) ()
#58 0x0000564a7e15fb2c in torch::full(c10::ArrayRef<long>, c10::Scalar const&, c10::TensorOptions) ()
#59 0x0000564a7e15c099 in main ()