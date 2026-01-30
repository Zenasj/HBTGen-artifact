def _validate_timeout(n):
    if n is unlimited:
        return -1
    else:
        return n