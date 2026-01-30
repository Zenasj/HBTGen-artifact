import torch
import torch.nn as nn
from torch.utils import cpp_extension
cuda_source = """
#include <c10/cuda/CUDACachingAllocator.h>
void my_fun(void)
{
    size_t temp_storage_bytes = 18446744073708433663UL;
    auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto temp_storage = caching_allocator.allocate(temp_storage_bytes);
    return;
}
"""
cpp_source = """
    void my_fun(void);
"""
module = torch.utils.cpp_extension.load_inline(
    name="cuda_test_extension",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions="my_fun",
    extra_cuda_cflags=["--extended-lambda"],
    verbose=True,
)
module.my_fun()
print('done')