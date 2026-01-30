cpp_source = """
    at::Tensor max_helper(at::Tensor self) {
    torch::Tensor y = torch::ones({3,1}, torch::kCUDA);
    torch::Tensor x  = torch::scalar_tensor({2.0});
    std::cout << torch::add(x, y) << std::endl;
    auto out = torch::max(x,y);
    std::cout <<out << std::endl;
    return out;

  }
"""
import torch
from torch.utils.cpp_extension import load_inline
module = load_inline(name='ext', cpp_sources=[cpp_source], functions=['max_helper'])
inp1=torch.randn(2,2, device="cuda")
module.max_helper(inp1)