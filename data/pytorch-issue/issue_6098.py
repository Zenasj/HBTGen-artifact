import torch

def _libcudnn():
    global lib, __cudnn_version
    if lib is None:
        if sys.platform == "win32":
            lib = find_cudnn_windows_lib()
        else:
            lib = ctypes.cdll.LoadLibrary(None)
        if hasattr(lib, 'cudnnGetErrorString'):
            lib.cudnnGetErrorString.restype = ctypes.c_char_p
            __cudnn_version = lib.cudnnGetVersion()
            compile_version = torch._C._cudnn_version()
            # Check that cuDNN major and minor versions match
            if (__cudnn_version // 100) != (compile_version // 100):
                raise RuntimeError(
                    'cuDNN version mismatch: PyTorch was compiled against {} '
                    'but linked against {}'.format(compile_version, __cudnn_version))
        else:
            lib = None
    return lib