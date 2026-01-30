import random

import torch
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
from codecs import encode

torch.serialization.add_safe_globals([encode, np.dtype, np.core.multiarray._reconstruct, np.ndarray])

with TemporaryDirectory() as tempdir:
    p = Path(tempdir)
    r2 = np.random.get_state()
    torch.save(r2, p / "r2.pkl")
    torch.load(p / "r2.pkl", weights_only=True)