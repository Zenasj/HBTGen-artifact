import torch

http_archive(
    name = "pytorch_cpu",
    build_file = "//tools:pytorch_cpu.BUILD",
    sha256 = "6b99edc046f37ad37a3b04dc86663895f10c362af09fdd10884f30360d0ba023",
    urls = [
        "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcpu.zip"
    ],
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "pytorch_cpu",
    includes = ["libtorch/include/torch/csrc/api/include", "libtorch/include"],
    hdrs = glob([
      "libtorch/include/**"
    ]),
    srcs = glob([
      "libtorch/lib/*",
    ]),
)

cc_library(
    name = "Test",
    srcs = ["Test.cpp"],
    deps = [
        "@pybind11",
        "@pytorch_cpu"
    ],
)