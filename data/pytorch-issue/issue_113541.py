class TensorParallelRNGTracker:
  ...
  def _distribute_region(self, spec: DTensorSpec):
    ...
    self._device_handle.set_rng_state(self.rng_states["tensor-parallel-rng"])
    try:
      yield
    finally:
      self.rng_states["tensor-parallel-rng"] = self._device_handle.get_rng_state()

import torch

device = torch._C._cuda_getDevice()
cuda_generator = torch.cuda.default_generators[device]

torch.cuda.manual_seed(0)
orig_state = cuda_generator.get_state()

# Function to execute RNG steps with optional state reset
def step(reset_state=False):
    torch.rand((3, 3), device=device)
    if reset_state:  # Restore the original state if requested
        cuda_generator.set_state(orig_state)
    torch.rand((3, 3), device=device)

print("Normal RNG, final offset should increment twice")
torch.cuda.manual_seed(0)
print("Initial offset =", cuda_generator.get_offset())
step(reset_state=False)
print("Final offset =", cuda_generator.get_offset())

print("\nRNG with state restoration, final offset should increment once")
torch.cuda.manual_seed(0)
orig_state = cuda_generator.get_state()
print("Initial offset =", cuda_generator.get_offset())
step(reset_state=True)
print("Final offset =", cuda_generator.get_offset())

print("\nCaptured with CUDA graph, RNG with state restoration.")
torch.cuda.manual_seed(0)
orig_state = cuda_generator.get_state()
print("Initial offset =", cuda_generator.get_offset())
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    step(reset_state=True)
g.replay()
print("Final offset =", cuda_generator.get_offset())

class TensorParallelRNGTracker:
  def __init__(self, device_type: str = "cuda"):
    self.rng_states["tensor-parallel-rng"] = self._device_handle.register_rng_state_with_index()
  
  def _distribute_region(self, spec: DTensorSpec):
    ...
    self._device_handle.set_rng_state(self.rng_states["tensor-parallel-rng"], use_index=True)
    try:
      yield
    finally:
      self.rng_states["tensor-parallel-rng"] = self._device_handle.get_rng_state(use_index=True)