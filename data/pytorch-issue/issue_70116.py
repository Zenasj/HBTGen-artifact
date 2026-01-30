class ProcessException(Exception):
    __slots__ = ["error_index", "error_pid"]

    def __init__(self, msg: str, error_index: int, pid: int):
        super().__init__(msg)
        self.error_index = error_index
        self.pid = pid

    def __reduce__(self):
        return (self.__class__, self.args + (self.error_index, self.pid), {})