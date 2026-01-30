import random

import os
import numpy as np
import torch.onnx
from model import Generator


if __name__ == "__main__":

        size = 512
        channel_multiplier = 1

        g = Generator(size, 512, 8, channel_multiplier=channel_multiplier)
        # g.load_state_dict(state_dict.get("g"))

        batch_size = {256: 16, 512: 9, 1024: 4}
        n_sample = batch_size.get(size, 25)
        z = np.random.RandomState(0).randn(n_sample, 512).astype("float32")

        gen_img = np.random.RandomState(0).random((n_sample, 3, size, size)).astype("float32")

        with torch.no_grad():
            # g.eval()

            filename = "model.onnx"
            torch.onnx.export(g,  # model being run
                              [torch.from_numpy(z)],  # model input (or a tuple for multiple inputs)
                              filename,  # where to save the model (can be a file or file-like object)
                              export_params=True,
                              opset_version=11)  # store the trained parameter weights inside the model