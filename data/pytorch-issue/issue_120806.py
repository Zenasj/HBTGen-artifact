import torch

def logging(func):
   # The logging_not_scriptable() decorator is not torch jit compatible
   wrapped = logging_not_scriptable(func)

   # Setting the __prepare_scriptable__ allows torch script to ignore the decorator
   # Useful for logging and typechecking decorators
   wrapped.__prepare_scriptable__ = lambda: func

@logging
def foo():
   ... # Valid scriptable code

foo_jit = torch.script.jit(foo) # The compiled version doesn't have the decorator