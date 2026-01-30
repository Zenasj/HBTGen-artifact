import caffe2.python.onnx.backend as backend
import numpy as np
rep = backend.prepare(model, device="CUDA:0") # or "CPU"