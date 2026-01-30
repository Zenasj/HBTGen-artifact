ffi = create_extension(
    '_ext.my_lib',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_compile_args=["-std=c99"],
    include_dirs=['C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\include'], # example
    library_dirs=['C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v9.0\\lib\\x64'], # example
    libraries=['ATen', '_C'] # Append cuda libaries when necessary, like cudart
)