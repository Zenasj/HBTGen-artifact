import numpy as np
import random

inputs = keras.Input(shape=(200,), dtype="int32")
# Embed each integer in a 128-dimensional vector
x = layers.Embedding(max_features, 256)(inputs)
# Add 2 bidirectional LSTMs
# x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
# x = layers.Bidirectional(layers.LSTM(64))(x)
x = layers.LSTM(64)(x)
# Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model_dir = '/data/danlu/tensorrt/resnet/lstm'
opt_model_dir = '/data/danlu/tensorrt/resnet/lstm_trt'
 
precision = "FP32"
max_workspace_size_bytes = 8000000000
conversion_params = tf_trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision,
                                                                  max_workspace_size_bytes=max_workspace_size_bytes,
                                                                  maximum_cached_engines=100)
converter = tf_trt.TrtGraphConverterV2(input_saved_model_dir=model_dir, conversion_params=conversion_params)
converter.convert()
 
 
def build_fn():
    Inp1 = np.random.randint(1, 200, (1, 200), dtype=np.int32)
    yield Inp1
 
 
converter.build(input_fn=build_fn)
converter.save(opt_model_dir)