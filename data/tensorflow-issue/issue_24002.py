load("//tensorflow/compiler/aot:tfcompile.bzl", "tf_library")
load("//tensorflow:tensorflow.bzl", "tf_copts")

# Use the tf_library macro to compile your graph into executable code.
tf_library(
    name = "punctuation_decode",
    cpp_class = "TensorflowXLA::PunctuationDecode",
    graph = "final.pb",
    config = "punctuation_decode.config.pbtxt",
    tfcompile_flags = "--target_features=+avx"
)

native.cc_binary(
  name = "libpunctuation.so",
  srcs = ["punctuation_decode_tfcompile_function.o", "punctuation_decode_tfcompile_metadata.o", "punctuation_decode.h"],
  linkshared=1,
  linkstatic=1,
  linkopts=["-fPIC"],
  deps = [
          # TODO(cwhipkey): only depend on kernel code that the model actually needed.
          # "//tensorflow/compiler/tf2xla/kernels:gather_op_kernel_float_int32",
          # "//tensorflow/compiler/tf2xla/kernels:gather_op_kernel_float_int64",
          "//tensorflow/compiler/tf2xla/kernels:index_ops_kernel_argmax_float_1d",
          "//tensorflow/compiler/tf2xla/kernels:index_ops_kernel_argmax_float_2d",
          # "//tensorflow/compiler/aot:runtime",
          # "//tensorflow/compiler/tf2xla:xla_local_runtime_context",
          "//tensorflow/compiler/xla/service/cpu:runtime_conv2d",
          "//tensorflow/compiler/xla/service/cpu:runtime_matmul",
          "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_conv2d",
          "//tensorflow/compiler/xla/service/cpu:runtime_single_threaded_matmul",
          "//tensorflow/compiler/xla:executable_run_options",
          "//third_party/eigen3",
          "//tensorflow/core:framework_lite",
          ],
  copts=tf_copts()
)