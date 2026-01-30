"-D__CUDA_NO_HALF_OPERATORS__"  # Since we have defined operators for the half type, we need to disable it using this flag. 
"--expt-relaxed-constexpr" # Enables `constexpr __host__` functions