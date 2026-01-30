AT_CUDA_CHECK(cudaGetLastError());

#define TORCH_CUDA_KERNEL_LAUNCH_CHECK() AT_CUDA_CHECK(cudaGetLastError())