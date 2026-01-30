py
class Close:
    def __init__(self, flag:bool, msg: str = "") -> None:
        self._flag = flag
        self._msg = msg
        
    def __bool__(self):
        return self._flag
    
    def __str__(self):
        return self._msg

py
def are_equal(*args, **kwargs):
    return not not_close_error_metas(*args, **kwargs)