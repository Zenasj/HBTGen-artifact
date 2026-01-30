import numpy as np
import random

with open("init_net.pb",'rb') as f:
    init_net = f.read()
with open("predict_net.pb",'rb') as f:
    predict_net = f.read()

init_net.RunAllOnGPU() # Added this line.
predict_net.RunAllOnGPU() # Added this line.

p = workspace.Predictor(init_net, predict_net)
img = np.random.randn(1, 3, 128, 128).astype(np.float32)
out = p.run([img])