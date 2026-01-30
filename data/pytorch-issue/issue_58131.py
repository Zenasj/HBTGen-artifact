import torch

from torch._utils import ExceptionWrapper

class TwoArgException(Exception):
    def __init__(self, msg, count): ...

# If you need a "real world" exception with two args, here's one from the stdlib:
# import asyncio
# TwoArgException = asyncio.exceptions.LimitOverrunError
# or if on Python 3.7, try:
# TwoArgException = asyncio.streams.LimitOverrunError

try:
    raise TwoArgException("oh no", 0)
except Exception as e:
    data = ExceptionWrapper(where="in a test case")

data.reraise()