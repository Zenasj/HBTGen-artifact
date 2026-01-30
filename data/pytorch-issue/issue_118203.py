import torch

class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    @classmethod
    def create_with_source(cls, value, source):
        return cls(
            value,
            source=source,
        )

    def __init__(self, fn, is_constant=False, **kwargs):
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False
        
        assert isinstance(
            fn, (types.FunctionType, torch.jit.ScriptFunction)
        ), f"expected FunctionType found {typestr(fn)} {fn}"