import torch

from torch.utils import cpp_extension


cpp_source = """
void my_fun(torch::Tensor x)
{
    std::cout << "x.std(0) " << x.std(0) << ", x.std(0, true) " << x.std(0, true) << std::endl;
    std::cout << "at::std(x, 0) " << at::std(x, 0) << ", at::std(x, 0, true) " << at::std(x, 0, true) << std::endl;
    return;
}
"""

module = torch.utils.cpp_extension.load_inline(
    name="std_test_extension",
    cpp_sources=cpp_source,
    functions="my_fun",
    verbose=True,
)

x = torch.randn(2, 3)
module.my_fun(x)

print('python ', x.std(0))