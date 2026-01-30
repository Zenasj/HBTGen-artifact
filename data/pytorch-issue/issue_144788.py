if sys.version_info < (3, 10):
    _Number = (bool, int, float)
else:
    _Number: TypeAlias = Number