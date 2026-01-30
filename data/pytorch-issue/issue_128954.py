import torch

def f(x):
  x = g1(x)
  x = g2(x)
  return g3(x)

def _check_triton_bf16_support(graph: GraphLowering) -> None:
    def warn_and_skip(device) -> None:
        from torch._dynamo.exc import SkipFrame

        device_props = torch.cuda.get_device_properties(device)
        warnings.warn(
            f"{device_props.name} does not support bfloat16 compilation natively, skipping"
        )
        raise SkipFrame("BF16 is not supported")