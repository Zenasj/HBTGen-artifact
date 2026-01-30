shell
#31 5185.6 /opt/pytorch/caffe2/operators/generate_proposals_op_util_nms_gpu_test.cc:449:29: warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
#31 5185.6    for (int itest = 0; itest < input_thresh.size(); ++itest) {
#31 5185.6                        ~~~~~~^~~~~~~~~~~~~~~~~~~~~
#31 5209.1 In file included from /opt/pytorch/third_party/ideep/mkl-dnn/third_party/oneDNN/include/dnnl.h:20:0,
#31 5209.1                  from /opt/pytorch/third_party/ideep/include/ideep/abstract_types.hpp:4,
#31 5209.1                  from /opt/pytorch/third_party/ideep/include/ideep.hpp:39,
#31 5209.1                  from /opt/pytorch/caffe2/ideep/ideep_utils.h:6,
#31 5209.1                  from /opt/pytorch/caffe2/python/pybind_state_ideep.cc:12:
#31 5209.1 /opt/pytorch/third_party/ideep/mkl-dnn/third_party/oneDNN/include/oneapi/dnnl/dnnl.h:23:10: fatal error: oneapi/dnnl/dnnl_config.h: No such file or directory
#31 5209.1  #include "oneapi/dnnl/dnnl_config.h"
#31 5209.1           ^~~~~~~~~~~~~~~~~~~~~~~~~~~
#31 5209.1 compilation terminated.
#31 5209.1 make[2]: *** [caffe2/CMakeFiles/caffe2_pybind11_state_gpu.dir/python/pybind_state_ideep.cc.o] Error 1
#31 5209.1 make[2]: *** Waiting for unfinished jobs....
#31 5225.2 make[1]: *** [caffe2/CMakeFiles/caffe2_pybind11_state_gpu.dir/all] Error 2
#31 5225.2 make[1]: *** Waiting for unfinished jobs....
#31 5258.6 /opt/pytorch/torch/csrc/cuda/shared/cudart.cpp: In function ‘void torch::cuda::shared::initCudartBindings(PyObject*)’:
#31 5258.6 /opt/pytorch/torch/csrc/cuda/shared/cudart.cpp:103:7: warning: ‘cudaError_t cudaProfilerInitialize(const char*, const char*, cudaOutputMode_t)’ is deprecated [-Wdeprecated-declarations]
#31 5258.6        cudaProfilerInitialize);
#31 5258.6        ^~~~~~~~~~~~~~~~~~~~~~
#31 5258.6 In file included from /opt/pytorch/torch/csrc/cuda/shared/cudart.cpp:5:0:
#31 5258.6 /usr/local/cuda/include/cuda_profiler_api.h:134:57: note: declared here
#31 5258.6  extern __CUDA_DEPRECATED __host__ cudaError_t CUDARTAPI cudaProfilerInitialize(const char *configFile,
#31 5258.6                                                          ^~~~~~~~~~~~~~~~~~~~~~
#31 5317.0 make: *** [all] Error 2