import warnings
import torch
import torch.utils.cpp_extension

source = '''
at::Tensor foo(at::Tensor x, int error_type) {
    std::ostringstream err_stream;
    err_stream << "Error with "  << x.type();

    TORCH_WARN(err_stream.str());
    return x.cos();
}
'''

t = torch.rand(2).double()

warn_mod = torch.utils.cpp_extension.load_inline(name='warnmod',
                                                    cpp_sources=[source],
                                                    functions=['foo'],
                                                    with_pytorch_error_handling=True)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("error")
    warn_mod.foo(t, 0)