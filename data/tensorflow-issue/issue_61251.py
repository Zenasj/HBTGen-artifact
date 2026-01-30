import tensorflow as tf
from tensorflow import keras

class Model(tf.keras.Model):
  def call(self, x):
    return (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    
model = Model()
model(tf.convert_to_tensor(5))
model.save(dir)

rt = tf.ragged.constant([[["hey"]]])
t = tf.constant([1])
loaded({"ragged_tensor_input": rt, "nested_tensor_input": {"nested_tensor1": t}})