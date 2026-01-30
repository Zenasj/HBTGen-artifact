import numpy as np

import os

import flax
import flax.linen
import flax.linen as nn
from flax.linen import fp8_ops

import jax
import jax.experimental
import jax.experimental.serialize_executable
import jax.numpy as jnp
from jax.experimental import topologies

h100_gpu_target_config = """
gpu_device_info {
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 49152
  shared_memory_per_core: 233472
  threads_per_core_limit: 2048
  core_count: 132
  fpus_per_core: 128
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: 3352320000000
  l2_cache_size: 52428800
  clock_rate_ghz: 1.98
  device_memory_size: 84929347584
  shared_memory_per_block_optin: 232448
  cuda_compute_capability {
    major: 9
  }
  registers_per_core_limit: 65536
  registers_per_block_limit: 65536
}
platform_name: "CUDA"
dnn_version_info {
  major: 9
  minor: 7
}
runtime_version {
  major: 12
  minor: 8
  patch: 0
}
device_description_str: "NVIDIA H100 80GB HBM3"
"""

# Fake not having a GPU on my machine
os.environ["XLA_FLAGS"] = " ".join([
  "--xla_gpu_enable_triton_gemm=false",
  "--xla_dump_to=./dump",
  "--xla_dump_hlo_as_text=true",
])
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_VMODULE"] = "gemm_rewriter=10"
# os.environ["TF_CPP_VMODULE"] = "cuda_executor=10,"

topology = topologies.get_topology_desc(
  platform="cuda",
  target_config=h100_gpu_target_config,
  topology="1x1x1",
)


model = nn.Dense(
  1024,
  dot_general_cls=fp8_ops.Fp8DirectDotGeneralOp
)

@jax.jit
def fn(params, x):
    return model.apply(params, x)

mesh = topologies.make_mesh(
  topology,
  (1,),
  ("devices",),
)
replicated_sharding = jax.sharding.NamedSharding(
  mesh,
  jax.sharding.PartitionSpec(
    "devices",
  ),
)
x = jax.ShapeDtypeStruct(
  (16, 2048, 1028),
  jnp.float32,
  sharding=replicated_sharding,
)
params = jax.eval_shape(
  model.init,
  jax.ShapeDtypeStruct((2,), jnp.uint32),
  x
)

inputs = [
  params, x
]

lowered = fn.lower(*inputs)  # <- fail here
compiled = lowered.compile()
print(compiled)
seriazied, in_tree, out_tree = jax.experimental.serialize_executable.serialize(compiled)