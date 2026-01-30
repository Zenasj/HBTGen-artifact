import numpy as np

# Prepare backend
print('Preparing backend...')
prep_t0 = time.time()
prepared_backend = caffe2.python.onnx.backend.prepare(model, device='CUDA:0')
print('Backend prepared in {} seconds'.format(time.time() - prep_t0))
# Run the ONNX model with Caffe2
print('Running inference..')
fwd_t0 = time.time()
outputs = prepared_backend.run(img_arr.astype(np.float32))[0]
print('Foward pass time: {} seconds'.format(time.time() - fwd_t0))