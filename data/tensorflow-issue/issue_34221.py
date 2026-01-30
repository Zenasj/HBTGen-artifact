from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow_datasets as tfds

## model don't work if uncomment following line
# tf.config.experimental_run_functions_eagerly(True)

TRAIN_DATASET = tfds.load(name="cifar10")['train']

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32,32, 3)),
    tf.keras.layers.Dense(10, activation='softmax'),
])

BATCH_SIZE = 50

model.compile(loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

train_set = TRAIN_DATASET.map(lambda item: (item['image'], item['label'])).batch(BATCH_SIZE)
model.fit(train_set, epochs = 5)

if not from_logits:
    if (isinstance(output, (ops.EagerTensor, variables_module.Variable)) or
        output.op.type != 'Softmax'):
      epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
      output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
      output = math_ops.log(output)
    else:
      # When softmax activation function is used for output operation, we
      # use logits from the softmax function directly to compute loss in order
      # to prevent collapsing zero when training.
      # See b/117284466
      assert len(output.op.inputs) == 1
      output = output.op.inputs[0]

import tensorflow as tf
import tensorflow_datasets as tfds

TRAIN_DATASET = tfds.load(name="cifar10")['train']

tf.config.experimental_run_functions_eagerly(True) # result won't be affected by eager/graph mode

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(32,32, 3)),
    tf.keras.layers.Dense(10), # no softmax in the layer
])

BATCH_SIZE = 50

model.compile(loss = tf.losses.SparseCategoricalCrossentropy(from_logits = True), # do softmax in loss function
              metrics=['accuracy'])

train_set = TRAIN_DATASET.map(lambda item: (item['image'], item['label'])).batch(BATCH_SIZE)
model.fit(train_set, epochs = 5)