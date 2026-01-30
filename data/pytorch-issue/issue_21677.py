import random

import time
import numpy as np
import onnxruntime as rt

im = np.random.rand(1, 3, 256, 384).astype('uint8')

sess = rt.InferenceSession("model.onnx")

t0 = time.time()
output = sess.run(['prob'], {'data': im})[0]
print (time.time() - t0)