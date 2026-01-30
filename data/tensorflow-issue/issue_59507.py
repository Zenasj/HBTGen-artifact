from tensorflow import keras

import tensorflow as tf

class CustomModel(tf.keras.Model):

  def __init__(self):
    super().__init__()


  def call(self, inputs, training):
    print('Tracing with', inputs)
    return inputs


model = CustomModel()


model.__call__ = tf.function(model.__call__)

print('Saving model...')

tf.saved_model.save(model, "saved_model", signatures=model.__call__.get_concrete_function(
            inputs={"x": tf.TensorSpec(shape=[1,], dtype=tf.float32)}, training=tf.TensorSpec(shape=None, dtype=tf.bool, name="training")
        ))

imported = tf.saved_model.load("saved_model")
imported(inputs={"x": [1.]}, training=True)

import tensorflow as tf

class CustomModel(tf.keras.Model):

  def __init__(self):
    super().__init__()


  def call(self, inputs, training):
    print('Tracing with', inputs)
    return inputs

  def __call__(self, *args, **kwargs):
        
        return super().__call__(*args, **kwargs)


model = CustomModel()