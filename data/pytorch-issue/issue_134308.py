import torch

py
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup
import pybind11

setup(
    name="repro",
    ext_modules=[CUDAExtension(
        "repro_kernels",
        ["repro.cu"],
        include_dirs=[pybind11.get_include()],
    )],
    cmdclass={"build_ext": BuildExtension},
)

py
from torch.profiler import profile, record_function, ProfilerActivity
import repro_kernels

print("Warming up")
for i in range(1000):
    repro_kernels.cudaGraphDemo()

print("Profiling")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    for i in range(100):
        print("Loop inside profiler", i)
        repro_kernels.cudaGraphDemo()

py
from torch.profiler import profile, record_function, ProfilerActivity
import repro_kernels

print("Warming up")
for i in range(1000):
    repro_kernels.cudaGraphDemo()

print("Profiling")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    print("Profiled something")

print("Outside the profiler")
for i in range(1000):
    repro_kernels.cudaGraphDemo()