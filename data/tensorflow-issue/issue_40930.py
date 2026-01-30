from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

# Activate mixed precision policy
mixed_precision_policy_arg: str = "mixed_float16"
if mixed_precision_policy_arg is not None:
    mixed_precision_policy = tf.keras.mixed_precision.experimental.Policy(
        mixed_precision_policy_arg)
    tf.keras.mixed_precision.experimental.set_policy(
        mixed_precision_policy)

# Size parameters for the model
img_height, img_width, img_depth = 224, 224, 3

# Model definition
inp = Input(shape=(int(img_height), int(img_width),
                    int(img_depth)))
mobilenet_model = MobileNetV2(input_shape=(int(img_height),
                                            int(img_width),
                                            int(img_depth)),
                                alpha=0.35,
                                include_top=False,
                                weights='imagenet',
                                input_tensor=inp,
                                pooling='avg')
out = Dense(1, activation='tanh')(mobilenet_model.output)

# Build the whole model.
model = Model(inputs=inp, outputs=out)

#Convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the TF Lite model.
with tf.io.gfile.GFile('./model.tflite', 'wb') as f:
    f.write(tflite_model)