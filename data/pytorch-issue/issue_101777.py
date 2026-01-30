from typing import Union
import torch

@torch.jit.script
def jit_test(
    ok: Union[int, float], bad: int | float
) -> None:
    pass

def test_union_optional_of_union_return(self):
        @torch.jit.script
        def fn() -> None | str | int:
            y: Optional[int | str] = "foo"
            return y