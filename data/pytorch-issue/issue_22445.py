import torch

torch.utils.cpp_extension.load_inline('test', 'int main() { return 0 }')