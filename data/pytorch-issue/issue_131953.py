py
import torch

@torch.compile
def f(x):
    return x.sin()

x = torch.randn(3, device="cuda")
f(x)

def profile_compile_time(
        cls, func: Any, phase_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        if not cls.enabled:
            return func(*args, **kwargs)