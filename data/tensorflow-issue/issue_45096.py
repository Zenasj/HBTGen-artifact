import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/bit/s-r50x1/1"),
])

model.build(input_shape=(None, 224, 224, 3))
model.save("/tmp")
# Success

class DummyModel(tf.keras.Model):

  def __init__(self):
    super(DummyModel, self).__init__()
    self.model = hub.KerasLayer("https://tfhub.dev/google/bit/s-r50x1/1")

  def call(self, inputs):
    return self.model(inputs)


model = DummyModel()
model.build((None, 224, 224, 3))
model.save("/tmp")
# Error