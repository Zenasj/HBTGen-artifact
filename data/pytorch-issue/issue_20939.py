import torch

a_scripted_module = torch.jit.script(an_nn_module)

# same behavior as before
@torch.jit.script
def some_fn():
    return 2

# just marks a function as ignored, if nothing
# ever calls it then this has no effect
@torch.jit.ignore
def some_fn2():
    return 2

# doesn't do anything, this function is already 
# the main entry point
@torch.jit.export
def some_fn3():
    return 2

class MyModule(torch.jit.ScriptModule):
  my_int_list: List[int]
 
  def __init__(self):
    self.my_int_list = [2]  # the type can be inferred since it has elements
    self.my_float_list: List[float] = []  # the type is specified manually
    self.my_int_list = [] # can specify the type out of line
    self.my_int = 2

class MyModule(torch.jit.ScriptModule):
  __annotations__ = {'my_int_list': List[float]}
 
  def __init__(self):
    self.my_int_list = [2]  # the type can be inferred since it has elements
    self.my_float_list = []  # the type is specified manually
    self.my_int = 2