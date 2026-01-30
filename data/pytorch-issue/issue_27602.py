import random

import numpy as np
import onnx
import caffe2.python.onnx.backend

batch = 1
channel = 3
image_h = 416
image_w = 416
img = np.random.random_sample([batch, channel, image_h, image_w]).astype(np.float32)

onnx_model = "tiny_yolov2/model.onnx"
model = onnx.load(onnx_model)

backend = caffe2.python.onnx.backend.prepare(model)
outputs = backend.run([img])

blobs = backend.workspace.Blobs()
caffe2_data = {k: backend.workspace.FetchBlob(k) for k in blobs}

layer_data = caffe2_data["convolution2d_1_output"]
print(layer_data.shape)