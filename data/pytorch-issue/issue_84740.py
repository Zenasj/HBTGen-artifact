#pragma comment(lib, "torch")
#pragma comment(lib, "torch_cpu")
#pragma comment(lib, "c10")
#pragma comment(lib, "c10_cuda")
#pragma comment(lib, "torch_cuda")
#pragma comment(lib, "torch_cuda_cu")
#pragma comment(lib, "torch_cuda_cpp")
#pragma comment(linker, "/INCLUDE:?warp_size@cuda@at@@YAHXZ")  // enable CUDA in libtorch
#pragma comment(linker, "/INCLUDE:?ignore_this_library_placeholder@@YAHXZ")