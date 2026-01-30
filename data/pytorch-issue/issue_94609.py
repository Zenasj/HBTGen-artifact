import torch.nn as nn

def clone_function(f: types.FunctionType):
    c = f.__code__
    new_code = types.CodeType(
        c.co_argcount, c.co_posonlyargcount, c.co_kwonlyargcount, c.co_nlocals,
        c.co_stacksize, c.co_flags, c.co_code, c.co_consts, c.co_names, c.co_varnames,
        c.co_filename, c.co_name, c.co_firstlineno, c.co_lnotab, c.co_freevars, c.co_cellvars)
    return types.FunctionType(new_code, f.__globals__, f.__name__, 
        argdefs = f.__defaults__,  closure = f.__closure__)

class CustomSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
        self.forward = types.MethodType(clone_function(self.forward), self)