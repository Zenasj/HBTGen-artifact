import torch

torch._dynamo.list_backends()
['aot_eager', 'aot_eager_decomp_partition', 'aot_torchxla_trace_once', 'aot_torchxla_trivial', 'aot_ts', 'aot_ts_nvfuser', 'cudagraphs', 'dynamo_accuracy_minifier_backend', 'dynamo_minifier_backend', 'eager', 'inductor', 'ipex', 'nvprims_aten', 'nvprims_nvfuser', 'onnxrt', 'tensorrt', 'torchxla_trace_once', 'torchxla_trivial', 'ts', 'tvm']

torch._dynamo.list_backends()
['aot_ts_nvfuser', 'cudagraphs', 'inductor', 'ipex', 'nvprims_nvfuser', 'onnxrt', 'tensorrt', 'tvm']

torch.compile(backend="torchxla_trivial")