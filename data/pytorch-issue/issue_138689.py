import random

import torch
import numpy as np

def test_controlflow():
    def is_scaled(val: np.ndarray) -> bool:
        return np.min(val) >= 0 and np.max(val) <= 1

    def process(val):
        # np.min(val) >= 0 and np.max(val) <= 1
        if is_scaled(val):
            val = val * 255
        val = val - np.mean(val, keepdims=True)
        return val

    val = np.random.rand(3, 224, 224)
    output = process(val)

    process_opt = torch.compile(process, dynamic=True)
    output_opt = process_opt(val)

    assert np.allclose(output, output_opt)

if __name__ == "__main__":
    test_controlflow()