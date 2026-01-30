# in _dynamo/
def reset():
    do_reset_stuff()

# in compiler/
def reset():
    do_reset_stuff() # As in copy paste the logic from _dynamo.reset

# in _dynamo/
import warnings
import inspect

def reset():
    function_name = inspect.currentframe().f_code.co_name
    warnings.warn(f"{function_name} is deprecated, use compiler.{function_name} instead", DeprecationWarning)
    return compiler.reset()

def pt2_enabled():
    if hasattr(torch, 'compile'):
        return True
    else:
        return False