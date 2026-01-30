if version.cuda in ["12.4", "12.6"]:
            with open("/proc/self/maps") as f:
                _maps = f.read()
            # libtorch_global_deps.so always depends in cudart, check if its installed via wheel
            if "nvidia/cuda_runtime/lib/libcudart.so" in _maps:
                # If all abovementioned conditions are met, preload nvjitlink
                _preload_cuda_deps("nvjitlink", "libnvJitLink.so.*[0-9]")