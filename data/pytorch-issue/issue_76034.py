class Works:
    """I won't stay with -OO :)"""

class Problem:
    """I am not a docstring anymore {}""".format(":(")

class WorkAround:
    """I am a {} docstring that won't stay with -OO :)"""

if WorkAround.__doc__:
    WorkAround.__doc__ = WorkAround.__doc__.format("formatted")