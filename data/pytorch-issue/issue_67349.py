cc_library(
    name = "libkineto",
    srcs = glob(
        [
            "src/*.cpp",
            "src/*.h",
        ],
    ),
    hdrs = glob([
        "include/*.h",
        "src/*.tpp",
    ]),
    copts = [
        "-DKINETO_NAMESPACE=libkineto",
        "-DHAS_CUPTI",
    ],
    includes = [
        "include",
    ],
    deps = [
        "@cuda//:cuda_headers",
        "@cuda//:cupti",
        "@cuda//:cupti_headers",
        "@cuda//:nvperf_host",
        "@cuda//:nvperf_target",
        "@fmt",
    ],
)

{
    "ph": "X", "cat": "Kernel", 
    "name": "cudnn::gemm::computeOffsetsKernel(cudnn::gemm::ComputeOffsetsParams)", "pid": 0, "tid": "stream 7",
    "ts": 1635202069792700, "dur": 4,
    "args": {
      "queued": 0, "device": 0, "context": 1,
      "stream": 7, "correlation": 34, "external id": 0,
      "registers per thread": 16,
      "shared memory": 0,
      "blocks per SM": -99,
      "warps per SM": -396,
      "grid": [99, 1, 1],
      "block": [128, 1, 1],
      "est. achieved occupancy %": -1237
    }
  },

{
    "ph": "X", "cat": "Operator", 
    "name": "ProfilerStep#10", "pid": 31383, "tid": "31383",
    "ts": 4687921561, "dur": 292460,
    "args": {
      "Device": 31383, "External id": 19719,
      "Trace name": "PyTorch Profiler", "Trace iteration": 0 
      
    }
  },