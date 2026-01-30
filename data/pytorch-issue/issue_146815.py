for f in _marked_safe_globals_set:
    module, name = f.__module__, f.__name__

for f in _marked_safe_globals_set:
    module, name = f.__module__, f.__qualname__

import torch
import numpy as np
def test_numpy() -> None:
    torch.save(
        dict(
            numpy_dtype=np.dtype('int64'),
        ),
        'numpy.pth'
    )
    # torch.serialization.add_safe_globals([np.dtype, np.dtypes.Int64DType])
    torch.load('numpy.pth')

test_numpy()