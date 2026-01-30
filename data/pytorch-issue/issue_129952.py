import torch

@functools.lru_cache(None)
def init_backend_registration():
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_cuda import CppWrapperCuda
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .halide import HalideScheduling
    from .triton import TritonScheduling
    from .wrapper import WrapperCodeGen

    if get_scheduling_for_device("cpu") is None:
        cpu_backends = {"cpp": CppScheduling, "halide": HalideScheduling}
        register_backend_for_device(
            "cpu",
            lambda *args, **kwargs: cpu_backends[config.cpu_backend](*args, **kwargs),
            WrapperCodeGen,
            CppWrapperCpu,
        )

    if get_scheduling_for_device("cuda") is None:
        # CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
        cuda_backends = {"triton": CUDACombinedScheduling, "halide": HalideScheduling}
        register_backend_for_device(
            "cuda",
            lambda *args, **kwargs: cuda_backends[config.cuda_backend](*args, **kwargs),
            WrapperCodeGen,
            CppWrapperCuda,
        )

    if get_scheduling_for_device("xpu") is None:
        register_backend_for_device("xpu", TritonScheduling, WrapperCodeGen)

@functools.lru_cache(None)
def init_backend_registration():
    from .cpp import CppScheduling
    from .cpp_wrapper_cpu import CppWrapperCpu
    from .cpp_wrapper_cuda import CppWrapperCuda
    from .cuda_combined_scheduling import CUDACombinedScheduling
    from .halide import HalideScheduling
    from .triton import TritonScheduling
    from .wrapper import WrapperCodeGen

    if get_scheduling_for_device("cpu") is None:
        cpu_backends = {"cpp": CppScheduling, "halide": HalideScheduling}
        register_backend_for_device(
            "cpu",
            lambda *args, **kwargs: cpu_backends[config.cpu_backend](*args, **kwargs),
            WrapperCodeGen,
            CppWrapperCpu,
        )

    if get_scheduling_for_device("cuda") is None:
        # CUDACombinedScheduling combines Triton and CUDA C++ scheduling for CUDA devices via delegation
        cuda_backends = {"triton": CUDACombinedScheduling, "halide": HalideScheduling}
        register_backend_for_device(
            "cuda",
            lambda *args, **kwargs: cuda_backends[config.cuda_backend](*args, **kwargs),
            WrapperCodeGen,
            CppWrapperCuda,
        )

    if get_scheduling_for_device("xpu") is None:
        register_backend_for_device("xpu", TritonScheduling, WrapperCodeGen)
    
    private_backend = torch._C._get_privateuse1_backend_name()                                                                                                                                                        
    if private_backend != "privateuseone":
        if get_scheduling_for_device(private_backend) is None:
            from torch.utils.backend_registration import _get_custom_mod_func
            try:    
                triton_scheduling = _get_custom_mod_func("TritonScheduling")
                wrapper_codegen = _get_custom_mod_func("WrapperCodeGen")
                if triton_scheduling and wrapper_codegen:
                    register_backend_for_device(private_backend, triton_scheduling, wrapper_codegen)
            except RuntimeError:
                pass