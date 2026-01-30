import torch

def test_union_optional_of_union_return(self):
        @torch.jit.script
        def fn() -> None | str | int:
            y: Optional[int | str] = "foo"
            return y

@torch.jit.script
def fn() -> Union[None, str, int]:
    y: Optional[Union[int, str]] = "foo"
    return y

@torch.jit.script
def fn() -> None | str | int:
    y: Optional[int | str] = "foo"
    return y

def test_union_optional_of_union_is_flattened(self):
        @torch.jit.script
        def fn(flag: int) -> str | int | None:
            y: int | str | None = "foo"
            if flag == 0:
                x: Optional[int | str] = y
            elif flag == 1:
                x: Optional[int | str] = 1
            else:
                x: Optional[int | str] = None
            return x