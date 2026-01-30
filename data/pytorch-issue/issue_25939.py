import torch.utils.cpp_extension

try:
    torch.utils.cpp_extension.load_inline(
            name="test", 
            cpp_sources="int main() { return 0 }",
            extra_include_paths=["/usr/include/python2.7"])
except RuntimeError as e:
    print(e.message)