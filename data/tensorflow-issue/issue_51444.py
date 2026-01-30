from tensorflow import keras
from tensorflow.keras import layers

class dummy(tf.keras.Model):
    def __init__(self):
        super(dummy, self).__init__()

    def call(self, inputs):
        lens = tf.map_fn(lambda x: len(x), inputs, fn_output_signature=tf.int32, name='get_lengths')
        tensored_input = inputs.to_tensor(0, shape=[None, 10])
        x = Lambda(lambda x: tf.cast(x, dtype=tf.float32))(tensored_input)
        outputs = tf.RaggedTensor.from_tensor(x, lengths=lens)
        return outputs

x_dataset = tf.data.Dataset.from_tensor_slices( tf.ragged.constant([[0,0,0,0,0],[0],[0,0,0],[0,0,0,0]]))
y_dataset = tf.data.Dataset.from_tensor_slices( tf.ragged.constant([[1,1,1,1,1],[1],[1,1,1],[1,1,1,1]]))
tensor_dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
# Create batches
batched_ds = tensor_dataset.batch(2, drop_remainder=True)

model = dummy()
model.compile(loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(batched_ds,epochs=2)

import tensorflow as tf
from tensorflow.keras.layers import Lambda

class dummy(tf.keras.Model):
    def __init__(self):
        super(dummy, self).__init__()

    def call(self, inputs):
        lens = tf.map_fn(lambda x: len(x), inputs, fn_output_signature=tf.int32, name='get_lengths')
        tensored_input = inputs.to_tensor(0, shape=[None, 10])
        x = Lambda(lambda x: tf.cast(x, dtype=tf.float32))(tensored_input)
        outputs = tf.RaggedTensor.from_tensor(x, lengths=lens)
        return outputs

x_dataset = tf.data.Dataset.from_tensor_slices( tf.ragged.constant([[0,0,0,0,0],[0],[0,0,0],[0,0,0,0]]))
y_dataset = tf.data.Dataset.from_tensor_slices( tf.ragged.constant([[1,1,1,1,1],[1],[1,1,1],[1,1,1,1]]))
tensor_dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
# Create batches
batched_ds = tensor_dataset.batch(2, drop_remainder=True)

model = dummy()
model.compile(loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(batched_ds,epochs=2)
# Accuracy should be 0 here obviously and it's not
print(history.history)