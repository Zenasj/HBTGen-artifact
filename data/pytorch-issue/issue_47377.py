CUDAExtension(
            'kernels',
            [
                'csrc/custom.cpp',
                'csrc/custom_kernel.cu'
            ],
            extra_link_args=['-lcusparse', '-l', 'cusparse'],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['--compiler-bindir=~/anaconda3/envs/rgnn_at_scale/bin/x86_64-conda-linux-gnu-cc']
            }
        )