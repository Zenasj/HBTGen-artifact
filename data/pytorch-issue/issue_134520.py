@triton.jit
def mykernel(
  param0,
  param1,
  param2,
  param3: tl.constexpr,   # autotuned
  param4,                 # non-constexpr
):
  ...