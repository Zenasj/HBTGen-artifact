import torch.nn as nn

P = ParamSpec('P')
T_co = TypeVar('T_co', covariant=True)

if TYPE_CHECKING:
    class Module(Generic[P, T_co]):
        def forward(self, *args: P.args, **kwargs: P.kwargs) -> T_co: ...
else:
    class Module:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

class MyModule(nn.Module[Tuple[Tensor], Tensor]):
    def forward(self, x: Tensor) -> Tensor:
        return x * 2

# Type checker catches errors:
class BadModule(nn.Module[Tuple[str], int]):  # Error: wrong types
    def forward(self, x: str) -> int:
        return 42