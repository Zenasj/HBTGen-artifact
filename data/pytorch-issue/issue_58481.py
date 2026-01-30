for sample in op.sample_inputs(...):
        ...

for ctx in op.sample_inputs(...):
        with ctx:
            ...

import contextlib
from typing import Iterator


@contextlib.contextmanager
def amend_error_msg(msg_fmtstr: str) -> Iterator[None]:
    try:
        yield
    except Exception as error:
        raise type(error)(msg_fmtstr.format(msg=str(error))) from error

a = 0
b = 1

with amend_error_msg(f"{{msg}}\n\nFailure happened for a={a} and b={b}"):
    assert a == b, "Values don't match!"

samples = op.sample_inputs() # this is a generator
first_sample = next(sample)
# the generator is left pending and will be closed at the end of the function and/or by the GC

for sample in op.sample_inputs(device, dtype, ...):
            test_logger.print('SampleInput: ', sample.name)
            ...